#!/usr/bin/env python3
"""Safely rename a Codex model-provider ID in the local thread store.

The default mode is a read-only preflight.  Applying the plan requires the
explicit ``--apply`` flag.  Rollout files are edited by replacing only the
JSON string containing ``payload.model_provider``; this keeps the rest of the
transcript byte-for-byte identical and lets us repair Codex's byte offsets for
paginated history.

This intentionally supports plain ``.jsonl`` rollouts only.  Codex may keep
cold rollouts as ``.jsonl.zst``; those are reported and block ``--apply`` so
the script never silently changes the meaning of stored offsets.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

try:
    import fcntl
except ImportError:  # pragma: no cover - Codex's supported Unix environments have fcntl.
    fcntl = None


PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
STATE_DB_RE = re.compile(r"^state_\d+\.sqlite$")
HISTORY_DB_RE = re.compile(r"^thread_history_\d+\.sqlite$")

# This is deliberately a byte regex.  It finds a JSON string value while
# preserving whitespace, key ordering, and every unrelated byte in the line.
MODEL_PROVIDER_FIELD_RE = re.compile(
    rb'(?<!\\)(?P<key>"model_provider"\s*:\s*)'
    rb'(?P<value>"(?:\\.|[^"\\])*")'
)

# Codex's config uses quoted TOML strings and conventional table headers.
CONFIG_ASSIGNMENT_RE = re.compile(
    r'(?m)^(?P<prefix>[ \t]*model_provider[ \t]*=[ \t]*)'
    r'(?P<quote>["\'])(?P<value>[^"\']*)(?P=quote)'
    r'(?P<tail>[ \t]*(?:#.*)?\r?$)'
)


class PlanError(RuntimeError):
    pass


@dataclasses.dataclass
class Issue:
    level: str
    message: str


@dataclasses.dataclass
class Replacement:
    start: int
    end: int
    old_bytes: bytes
    new_bytes: bytes

    @property
    def delta(self) -> int:
        return len(self.new_bytes) - len(self.old_bytes)


@dataclasses.dataclass
class Rollout:
    path: Path
    kind: str
    compressed: bool = False
    provider: str | None = None
    thread_id: str | None = None
    replacement: Replacement | None = None
    size: int = 0
    error: str | None = None


@dataclasses.dataclass
class StateRow:
    db: Path
    thread_id: str
    rollout_path: str
    history_mode: str


@dataclasses.dataclass
class StateDb:
    path: Path
    rows: list[StateRow] = dataclasses.field(default_factory=list)
    has_threads: bool = False
    error: str | None = None


@dataclasses.dataclass
class OffsetUpdate:
    table: str
    thread_id: str
    turn_id: str | None
    column: str
    old_value: int
    new_value: int


@dataclasses.dataclass
class HistoryDb:
    path: Path
    updates: list[OffsetUpdate] = dataclasses.field(default_factory=list)
    projection_threads: set[str] = dataclasses.field(default_factory=set)
    turn_threads: set[str] = dataclasses.field(default_factory=set)
    error: str | None = None


@dataclasses.dataclass
class ConfigPlan:
    path: Path
    exists: bool = False
    old_text: str = ""
    new_text: str = ""
    assignment_count: int = 0
    header_count: int = 0
    target_header_exists: bool = False
    error: str | None = None

    @property
    def changed(self) -> bool:
        return self.exists and self.old_text != self.new_text


@dataclasses.dataclass
class Plan:
    codex_home: Path
    config: ConfigPlan
    rollouts: list[Rollout]
    state_dbs: list[StateDb]
    history_dbs: list[HistoryDb]
    source: str
    target: str
    issues: list[Issue] = dataclasses.field(default_factory=list)

    @property
    def matching_rollouts(self) -> list[Rollout]:
        return [r for r in self.rollouts if r.replacement is not None]

    @property
    def changed_files(self) -> list[Path]:
        paths = [r.path for r in self.matching_rollouts]
        paths.extend(db.path for db in self.state_dbs if db.rows)
        paths.extend(db.path for db in self.history_dbs if db.updates)
        if self.config.changed:
            paths.append(self.config.path)
        return list(dict.fromkeys(paths))


def default_codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def validate_provider_id(value: str, label: str) -> None:
    if not PROVIDER_ID_RE.fullmatch(value):
        raise PlanError(
            f"{label} provider ID {value!r} is not safe for Codex TOML headers; "
            "use letters, digits, '-' or '_' and start with a letter/digit"
        )


def canonical_plain_path(path: Path) -> Path:
    if path.name.endswith(".jsonl.zst"):
        return path.with_name(path.name[:-4])
    return path


def discover_rollouts(codex_home: Path) -> list[tuple[Path, str, bool]]:
    """Return one physical path per logical rollout, preferring plain JSONL."""

    candidates: dict[Path, tuple[Path | None, Path | None, str]] = {}
    for kind, root_name in (("active", "sessions"), ("archived", "archived_sessions")):
        root = codex_home / root_name
        if not root.exists():
            continue
        if not root.is_dir():
            raise PlanError(f"expected directory but found {root}")
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            if path.name.endswith(".jsonl.zst"):
                logical = canonical_plain_path(path)
                plain, compressed, _ = candidates.get(logical, (None, None, kind))
                candidates[logical] = (plain, path, kind)
            elif path.name.endswith(".jsonl"):
                logical = path
                plain, compressed, _ = candidates.get(logical, (None, None, kind))
                candidates[logical] = (path, compressed, kind)

    result: list[tuple[Path, str, bool]] = []
    for logical in sorted(candidates):
        plain, compressed, kind = candidates[logical]
        if plain is not None:
            result.append((plain, kind, False))
        elif compressed is not None:
            result.append((compressed, kind, True))
    return result


def read_first_nonempty_line(path: Path) -> tuple[int, bytes] | None:
    offset = 0
    with path.open("rb") as handle:
        while True:
            line_start = offset
            line = handle.readline()
            if not line:
                return None
            offset += len(line)
            if line.strip():
                return line_start, line


def scan_rollout(path: Path, kind: str, compressed: bool, source: str, target: str) -> Rollout:
    item = Rollout(path=path, kind=kind, compressed=compressed)
    try:
        item.size = path.stat().st_size
    except OSError as exc:
        item.error = f"cannot stat: {exc}"
        return item
    if compressed:
        item.error = "compressed .jsonl.zst rollout is not supported for safe offset repair"
        return item

    try:
        first = read_first_nonempty_line(path)
        if first is None:
            return item
        line_start, line = first
        parsed = json.loads(line.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        item.error = f"cannot parse first metadata line: {exc}"
        return item

    if not isinstance(parsed, dict) or parsed.get("type") != "session_meta":
        return item
    payload = parsed.get("payload")
    if not isinstance(payload, dict):
        item.error = "session_meta payload is not an object"
        return item
    provider = payload.get("model_provider")
    item.provider = provider if isinstance(provider, str) else None
    thread_id = payload.get("id", payload.get("session_id"))
    item.thread_id = thread_id if isinstance(thread_id, str) else None
    if provider != source:
        return item

    matches = []
    for match in MODEL_PROVIDER_FIELD_RE.finditer(line):
        try:
            value = json.loads(match.group("value").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if value == source:
            matches.append(match)
    if len(matches) != 1:
        item.error = (
            "could not identify exactly one payload.model_provider JSON token "
            f"(found {len(matches)})"
        )
        return item

    match = matches[0]
    old_bytes = match.group("value")
    new_bytes = json.dumps(target, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    replacement = Replacement(
        start=line_start + match.start("value"),
        end=line_start + match.end("value"),
        old_bytes=old_bytes,
        new_bytes=new_bytes,
    )
    candidate_line = line[: match.start("value")] + new_bytes + line[match.end("value") :]
    try:
        candidate = json.loads(candidate_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        item.error = f"provider replacement would make metadata invalid JSON: {exc}"
        return item
    if not isinstance(candidate, dict) or not isinstance(candidate.get("payload"), dict):
        item.error = "provider replacement changed metadata shape"
        return item
    if candidate["payload"].get("model_provider") != target:
        item.error = "provider replacement did not update payload.model_provider"
        return item
    item.replacement = replacement
    return item


def config_header_pattern(provider: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?m)^(?P<prefix>[ \t]*\[\[?model_providers\.)"
        rf"{re.escape(provider)}(?=(?:\.|\]|\s))"
    )


def plan_config(path: Path, source: str, target: str) -> ConfigPlan:
    plan = ConfigPlan(path=path)
    if not path.exists():
        return plan
    plan.exists = True
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            plan.old_text = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        plan.error = f"cannot read config: {exc}"
        return plan

    if config_header_pattern(target).search(plan.old_text):
        plan.target_header_exists = True
        plan.error = f"config already contains [model_providers.{target}]"
        return plan

    old_header = config_header_pattern(source)
    plan.header_count = len(old_header.findall(plan.old_text))

    def replace_assignment(match: re.Match[str]) -> str:
        if match.group("value") != source:
            return match.group(0)
        plan.assignment_count += 1
        return (
            match.group("prefix")
            + match.group("quote")
            + target
            + match.group("quote")
            + match.group("tail")
        )

    rewritten = CONFIG_ASSIGNMENT_RE.sub(replace_assignment, plan.old_text)
    rewritten, header_count = old_header.subn(rf"\g<prefix>{target}", rewritten)
    # The header count is counted on the source text, while subn sees the
    # post-assignment text; they are equivalent and this keeps the result clear.
    if header_count:
        plan.header_count = header_count
    plan.new_text = rewritten

    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        tomllib.loads(plan.new_text)
    except Exception as exc:
        plan.error = f"rewritten config does not parse as TOML: {exc}"
    return plan


def sqlite_uri(path: Path) -> str:
    return f"file:{quote(str(path), safe='/')}?mode=ro"


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
    ).fetchone()
    return row is not None


def table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def inspect_state_dbs(codex_home: Path, source: str) -> list[StateDb]:
    result: list[StateDb] = []
    for path in sorted(p for p in codex_home.iterdir() if p.is_file() and STATE_DB_RE.fullmatch(p.name)):
        item = StateDb(path=path)
        try:
            connection = sqlite3.connect(sqlite_uri(path), uri=True, timeout=2.0)
            connection.execute("PRAGMA busy_timeout=2000")
            if not table_exists(connection, "threads"):
                connection.close()
                result.append(item)
                continue
            item.has_threads = True
            columns = table_columns(connection, "threads")
            needed = {"id", "rollout_path", "model_provider", "history_mode"}
            if not needed.issubset(columns):
                item.error = f"threads table is missing columns: {sorted(needed - columns)}"
            else:
                rows = connection.execute(
                    "SELECT id, rollout_path, model_provider, history_mode "
                    "FROM threads WHERE model_provider = ?",
                    (source,),
                ).fetchall()
                item.rows = [
                    StateRow(
                        db=path,
                        thread_id=str(row[0]),
                        rollout_path=str(row[1]),
                        history_mode=str(row[3]),
                    )
                    for row in rows
                ]
            connection.close()
        except (OSError, sqlite3.Error) as exc:
            item.error = f"cannot inspect SQLite database: {exc}"
        result.append(item)
    return result


def resolve_state_rollout(codex_home: Path, stored_path: str) -> Path | None:
    path = Path(stored_path)
    if not path.is_absolute():
        path = codex_home / path
    if path.exists() and path.is_file():
        return path
    if path.name.endswith(".jsonl"):
        compressed = path.with_name(path.name + ".zst")
        if compressed.exists() and compressed.is_file():
            return compressed
    return None


def adjust_offset(value: int, replacement: Replacement, old_size: int) -> int:
    if value < 0 or value > old_size:
        raise PlanError(f"stored offset {value} is outside rollout size {old_size}")
    if value < replacement.start:
        new_value = value
    elif value < replacement.end:
        raise PlanError(
            f"stored offset {value} falls inside provider token range "
            f"[{replacement.start}, {replacement.end})"
        )
    else:
        new_value = value + replacement.delta
    new_size = old_size + replacement.delta
    if new_value < 0 or new_value > new_size:
        raise PlanError(f"adjusted offset {new_value} is outside new rollout size {new_size}")
    return new_value


def inspect_history_dbs(
    codex_home: Path,
    affected: dict[str, Rollout],
    issues: list[Issue],
) -> list[HistoryDb]:
    result: list[HistoryDb] = []
    ids = sorted(affected)
    if not ids:
        return result
    for path in sorted(
        p for p in codex_home.iterdir() if p.is_file() and HISTORY_DB_RE.fullmatch(p.name)
    ):
        item = HistoryDb(path=path)
        try:
            connection = sqlite3.connect(sqlite_uri(path), uri=True, timeout=2.0)
            connection.execute("PRAGMA busy_timeout=2000")
            has_projection = table_exists(connection, "thread_history_projection_state")
            has_turns = table_exists(connection, "thread_turns")
            if has_projection:
                columns = table_columns(connection, "thread_history_projection_state")
                if {"thread_id", "next_rollout_byte_offset"}.issubset(columns):
                    placeholders = ",".join("?" for _ in ids)
                    rows = connection.execute(
                        "SELECT thread_id, next_rollout_byte_offset "
                        f"FROM thread_history_projection_state WHERE thread_id IN ({placeholders})",
                        ids,
                    ).fetchall()
                    for thread_id, old_value in rows:
                        item.projection_threads.add(str(thread_id))
                        rollout = affected.get(str(thread_id))
                        if rollout is None or rollout.replacement is None:
                            continue
                        try:
                            new_value = adjust_offset(
                                int(old_value), rollout.replacement, rollout.size
                            )
                        except PlanError as exc:
                            issues.append(Issue("error", f"{path}: thread {thread_id}: {exc}"))
                            continue
                        if new_value != old_value:
                            item.updates.append(
                                OffsetUpdate(
                                    table="thread_history_projection_state",
                                    thread_id=str(thread_id),
                                    turn_id=None,
                                    column="next_rollout_byte_offset",
                                    old_value=int(old_value),
                                    new_value=new_value,
                                )
                            )
            if has_turns:
                columns = table_columns(connection, "thread_turns")
                needed = {"thread_id", "turn_id", "rollout_byte_offset", "rollout_end_byte_offset"}
                if needed.issubset(columns):
                    placeholders = ",".join("?" for _ in ids)
                    rows = connection.execute(
                        "SELECT thread_id, turn_id, rollout_byte_offset, rollout_end_byte_offset "
                        f"FROM thread_turns WHERE thread_id IN ({placeholders})",
                        ids,
                    ).fetchall()
                    for thread_id, turn_id, start_value, end_value in rows:
                        item.turn_threads.add(str(thread_id))
                        rollout = affected.get(str(thread_id))
                        if rollout is None or rollout.replacement is None:
                            continue
                        for column, old_value in (
                            ("rollout_byte_offset", start_value),
                            ("rollout_end_byte_offset", end_value),
                        ):
                            if old_value is None:
                                continue
                            try:
                                new_value = adjust_offset(
                                    int(old_value), rollout.replacement, rollout.size
                                )
                            except PlanError as exc:
                                issues.append(Issue("error", f"{path}: thread {thread_id}: {exc}"))
                                continue
                            if new_value != old_value:
                                item.updates.append(
                                    OffsetUpdate(
                                        table="thread_turns",
                                        thread_id=str(thread_id),
                                        turn_id=str(turn_id),
                                        column=column,
                                        old_value=int(old_value),
                                        new_value=new_value,
                                    )
                                )
            connection.close()
        except (OSError, sqlite3.Error) as exc:
            item.error = f"cannot inspect SQLite history database: {exc}"
        result.append(item)
    return result


def build_plan(codex_home: Path, config_path: Path, source: str, target: str) -> Plan:
    validate_provider_id(source, "source")
    validate_provider_id(target, "target")
    if source == target:
        raise PlanError("source and target provider IDs must differ")
    codex_home = codex_home.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    config = plan_config(config_path, source, target)
    rollouts: list[Rollout] = []
    issues: list[Issue] = []
    try:
        discovered = discover_rollouts(codex_home)
    except (OSError, PlanError) as exc:
        discovered = []
        issues.append(Issue("error", str(exc)))
    for path, kind, compressed in discovered:
        item = scan_rollout(path, kind, compressed, source, target)
        if item.error:
            issues.append(Issue("error", f"{path}: {item.error}"))
        rollouts.append(item)
    if config.error:
        issues.append(Issue("error", f"{config.path}: {config.error}"))

    state_dbs = inspect_state_dbs(codex_home, source)
    for database in state_dbs:
        if database.error:
            issues.append(Issue("error", f"{database.path}: {database.error}"))

    has_store_matches = bool(
        any(item.replacement is not None for item in rollouts)
        or any(database.rows for database in state_dbs)
    )
    if has_store_matches and not config.exists:
        issues.append(
            Issue(
                "error",
                f"{config.path}: config is missing; refusing to rename history without a configured target provider",
            )
        )
    elif has_store_matches and config.assignment_count == 0 and config.header_count == 0:
        issues.append(
            Issue(
                "error",
                f"{config.path}: no {source!r} provider reference was found; pass the correct Codex config path",
            )
        )

    by_thread: dict[str, Rollout] = {}
    by_plain_path = {canonical_plain_path(item.path).resolve(): item for item in rollouts}
    for item in rollouts:
        if item.replacement is not None and item.thread_id:
            if item.thread_id in by_thread and by_thread[item.thread_id].path != item.path:
                issues.append(Issue("error", f"duplicate rollout metadata thread ID {item.thread_id}"))
            by_thread[item.thread_id] = item
        elif item.replacement is not None:
            issues.append(Issue("error", f"{item.path}: matching session metadata has no thread ID"))

    for database in state_dbs:
        for row in database.rows:
            resolved = resolve_state_rollout(codex_home, row.rollout_path)
            if resolved is None:
                issues.append(
                    Issue("error", f"{database.path}: thread {row.thread_id} points to missing rollout {row.rollout_path}")
                )
                continue
            item = by_plain_path.get(canonical_plain_path(resolved).resolve())
            if item is None:
                issues.append(
                    Issue("error", f"{database.path}: thread {row.thread_id} rollout is outside discovered session roots: {resolved}")
                )
                continue
            if item.compressed:
                issues.append(Issue("error", f"{database.path}: thread {row.thread_id} uses compressed rollout {resolved}"))
            elif item.provider != source:
                issues.append(
                    Issue("error", f"{database.path}: thread {row.thread_id} provider disagrees with rollout metadata ({item.provider!r})")
                )
            elif item.thread_id and item.thread_id != row.thread_id:
                issues.append(
                    Issue("error", f"{database.path}: thread {row.thread_id} does not match rollout metadata ID {item.thread_id}")
                )

    history_dbs = inspect_history_dbs(codex_home, by_thread, issues)
    for database in history_dbs:
        if database.error:
            issues.append(Issue("error", f"{database.path}: {database.error}"))

    paginated_ids = {
        row.thread_id
        for database in state_dbs
        for row in database.rows
        if row.history_mode == "paginated" and row.thread_id in by_thread
    }
    projected_ids = {
        thread_id
        for database in history_dbs
        for thread_id in database.projection_threads
    }
    missing_projection_ids = sorted(paginated_ids - projected_ids)
    if missing_projection_ids:
        issues.append(
            Issue(
                "error",
                "paginated threads have no thread_history projection row to repair: "
                + ", ".join(missing_projection_ids[:12])
                + (" ..." if len(missing_projection_ids) > 12 else ""),
            )
        )

    # A matching compressed rollout cannot be scanned, so it is always a hard
    # blocker.  Unrelated compressed rollouts are also blocked because the
    # script cannot prove that the logical store is fully understood.
    if any(item.compressed for item in rollouts):
        issues.append(Issue("error", "one or more compressed .jsonl.zst rollouts were discovered; materialize them before applying"))

    if not codex_home.exists():
        issues.append(Issue("error", f"Codex home does not exist: {codex_home}"))
    if not any(item.replacement for item in rollouts) and not any(db.rows for db in state_dbs) and not config.changed:
        issues.append(Issue("warning", f"no references to provider {source!r} were found in the supported store"))

    return Plan(
        codex_home=codex_home,
        config=config,
        rollouts=rollouts,
        state_dbs=state_dbs,
        history_dbs=history_dbs,
        source=source,
        target=target,
        issues=issues,
    )


def print_plan(plan: Plan, dry_run: bool = True) -> None:
    active = sum(item.replacement is not None and item.kind == "active" for item in plan.rollouts)
    archived = sum(item.replacement is not None and item.kind == "archived" for item in plan.rollouts)
    compressed = sum(item.compressed for item in plan.rollouts)
    state_rows = sum(len(item.rows) for item in plan.state_dbs)
    history_updates = sum(len(item.updates) for item in plan.history_dbs)
    delta_bytes = sum(item.replacement.delta for item in plan.matching_rollouts if item.replacement)
    mode = "DRY RUN (read-only)" if dry_run else "APPLY"
    print(f"{mode}: Codex provider {plan.source!r} -> {plan.target!r}")
    print(f"Codex home: {plan.codex_home}")
    print(
        f"Rollouts: {len(plan.rollouts)} discovered; {active} active + {archived} archived match; "
        f"{compressed} compressed; total byte delta {delta_bytes:+d}"
    )
    print(f"State DB rows to update: {state_rows}")
    print(f"Paginated-history offset cells to update: {history_updates}")
    print(
        f"Config: {plan.config.path} "
        f"({plan.config.assignment_count} model_provider assignments, {plan.config.header_count} provider headers)"
        if plan.config.exists
        else f"Config: {plan.config.path} (missing; no config edit)"
    )
    print(f"Files that would change: {len(plan.changed_files)}")
    for issue in plan.issues:
        print(f"{issue.level.upper()}: {issue.message}")
    if not plan.issues:
        print("Preflight: ready; no safety blockers found.")


def copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def patch_rollout(rollout: Rollout) -> None:
    if rollout.replacement is None:
        return
    replacement = rollout.replacement
    original_stat = rollout.path.stat()
    fd, temporary = tempfile.mkstemp(prefix=f".{rollout.path.name}.", dir=rollout.path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as output, rollout.path.open("rb") as source:
            remaining = replacement.start
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise PlanError(f"{rollout.path}: file ended before replacement offset")
                output.write(chunk)
                remaining -= len(chunk)
            current = source.read(len(replacement.old_bytes))
            if current != replacement.old_bytes:
                raise PlanError(f"{rollout.path}: metadata changed after preflight; refusing to patch")
            output.write(replacement.new_bytes)
            shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        shutil.copymode(rollout.path, temporary_path)
        os.replace(temporary_path, rollout.path)
        os.utime(rollout.path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    finally:
        temporary_path.unlink(missing_ok=True)


def backup_relative_path(codex_home: Path, path: Path, index: int) -> Path:
    try:
        relative = path.resolve().relative_to(codex_home)
        return Path("files") / relative
    except ValueError:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        return Path("external") / f"{index:03d}-{digest}-{path.name}"


def sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_uri(source), uri=True, timeout=5.0)
    try:
        target = sqlite3.connect(destination)
        try:
            connection.backup(target)
            target.commit()
        finally:
            target.close()
    finally:
        connection.close()


def make_backup(plan: Plan, backup_dir: Path) -> list[tuple[Path, Path, str]]:
    backup_dir.mkdir(parents=True, exist_ok=False)
    targets: list[tuple[Path, str]] = []
    targets.extend((item.path, "file") for item in plan.matching_rollouts)
    targets.extend((item.path, "sqlite") for item in plan.state_dbs if item.rows)
    targets.extend((item.path, "sqlite") for item in plan.history_dbs if item.updates)
    if plan.config.changed:
        targets.append((plan.config.path, "file"))
    unique: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for path, kind in targets:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append((path, kind))

    manifest: list[dict[str, str]] = []
    entries: list[tuple[Path, Path, str]] = []
    for index, (source, kind) in enumerate(unique):
        relative_backup = backup_relative_path(plan.codex_home, source, index)
        destination = backup_dir / relative_backup
        if kind == "sqlite":
            sqlite_backup(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        entries.append((source, destination, kind))
        manifest.append(
            {
                "source": str(source),
                "backup": str(relative_backup),
                "kind": kind,
            }
        )
    (backup_dir / "manifest.json").write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "codex_home": str(plan.codex_home),
                "source_provider": plan.source,
                "target_provider": plan.target,
                "files": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return entries


def restore_backup(entries: list[tuple[Path, Path, str]]) -> list[str]:
    errors: list[str] = []
    for source, backup, kind in reversed(entries):
        try:
            if kind == "sqlite":
                for suffix in ("-wal", "-shm"):
                    source.with_name(source.name + suffix).unlink(missing_ok=True)
            copy_file_atomic(backup, source)
        except OSError as exc:
            errors.append(f"{source}: {exc}")
    return errors


def apply_state_updates(database: StateDb, source: str, target: str) -> None:
    if not database.rows:
        return
    connection = sqlite3.connect(database.path, timeout=5.0)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "UPDATE threads SET model_provider = ? WHERE model_provider = ?",
            (target, source),
        )
        connection.commit()
    finally:
        connection.close()


def apply_history_updates(database: HistoryDb) -> None:
    if not database.updates:
        return
    connection = sqlite3.connect(database.path, timeout=5.0)
    try:
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN IMMEDIATE")
        for update in database.updates:
            if update.table == "thread_history_projection_state":
                cursor = connection.execute(
                    "UPDATE thread_history_projection_state SET next_rollout_byte_offset = ? "
                    "WHERE thread_id = ? AND next_rollout_byte_offset = ?",
                    (update.new_value, update.thread_id, update.old_value),
                )
            else:
                cursor = connection.execute(
                    f"UPDATE thread_turns SET {update.column} = ? "
                    f"WHERE thread_id = ? AND turn_id = ? AND {update.column} = ?",
                    (update.new_value, update.thread_id, update.turn_id, update.old_value),
                )
            if cursor.rowcount != 1:
                raise PlanError(
                    f"{database.path}: expected to update one {update.table}.{update.column} "
                    f"cell for {update.thread_id}, changed {cursor.rowcount}"
                )
        connection.commit()
    finally:
        connection.close()


def atomic_write_text(path: Path, text: str) -> None:
    original_stat = path.stat()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        shutil.copymode(path, temporary_path)
        os.replace(temporary_path, path)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    finally:
        temporary_path.unlink(missing_ok=True)


@contextlib.contextmanager
def held_codex_locks(codex_home: Path) -> Iterator[None]:
    if fcntl is None:
        raise PlanError("this apply path requires POSIX file locking")
    handles = []
    try:
        maintenance = codex_home / ".tmp" / "rollout-maintenance.lock"
        coordination = codex_home / "thread-writer-locks" / ".coordination.lock"
        for path in (maintenance, coordination):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise PlanError(f"active Codex maintenance/writer activity holds {path}: {exc}") from exc
            handles.append(handle)

        writer_dir = codex_home / "thread-writer-locks"
        for path in sorted(writer_dir.glob("*.lock")):
            if path.name == ".coordination.lock":
                continue
            handle = path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                raise PlanError(f"active Codex thread writer holds {path}: {exc}") from exc
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()


def default_backup_dir(codex_home: Path) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return codex_home / f"codex-provider-rename-backup-{stamp}-{uuid.uuid4().hex[:8]}"


def apply_plan(plan: Plan, backup_dir: Path) -> None:
    entries = make_backup(plan, backup_dir)
    try:
        for rollout in plan.matching_rollouts:
            patch_rollout(rollout)
        for database in plan.state_dbs:
            apply_state_updates(database, plan.source, plan.target)
        for database in plan.history_dbs:
            apply_history_updates(database)
        if plan.config.changed:
            atomic_write_text(plan.config.path, plan.config.new_text)
    except Exception:
        rollback_errors = restore_backup(entries)
        if rollback_errors:
            raise PlanError(
                f"apply failed; automatic rollback also had errors: {'; '.join(rollback_errors)}; "
                f"backup retained at {backup_dir}"
            )
        raise PlanError(f"apply failed; changes were rolled back from {backup_dir}: {sys.exc_info()[1]}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight or safely rename a Codex model_provider ID (dry-run by default)."
    )
    parser.add_argument("--from", dest="source", required=True, help="existing provider ID")
    parser.add_argument("--to", dest="target", required=True, help="new provider ID")
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=default_codex_home(),
        help="Codex home (default: CODEX_HOME or ~/.codex)",
    )
    parser.add_argument("--config", type=Path, help="config.toml path (default: <codex-home>/config.toml)")
    parser.add_argument("--apply", action="store_true", help="write changes; without this flag the run is read-only")
    parser.add_argument("--backup-dir", type=Path, help="backup directory for --apply")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    codex_home = args.codex_home.expanduser()
    config_path = args.config.expanduser() if args.config else codex_home / "config.toml"
    try:
        plan = build_plan(codex_home, config_path, args.source, args.target)
    except (PlanError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print_plan(plan, dry_run=not args.apply)
    blockers = [issue for issue in plan.issues if issue.level == "error"]
    if blockers:
        if args.apply:
            print("Apply refused because preflight has safety blockers.", file=sys.stderr)
        return 2
    if not args.apply:
        print("No files, databases, or config were changed.")
        return 0

    backup_dir = args.backup_dir.expanduser() if args.backup_dir else default_backup_dir(plan.codex_home)
    try:
        with held_codex_locks(plan.codex_home):
            locked_plan = build_plan(plan.codex_home, config_path, args.source, args.target)
            print_plan(locked_plan, dry_run=False)
            blockers = [issue for issue in locked_plan.issues if issue.level == "error"]
            if blockers:
                print("Apply refused after lock-protected recheck.", file=sys.stderr)
                return 2
            apply_plan(locked_plan, backup_dir)
    except (PlanError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Applied successfully. Backup retained at {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
