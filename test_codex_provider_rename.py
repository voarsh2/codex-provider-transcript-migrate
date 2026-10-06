#!/usr/bin/env python3
"""Synthetic tests for codex_provider_rename.py.

These tests use temporary Codex homes and never touch the user's real store.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("codex_provider_rename.py")


def write_rollout(path: Path, thread_id: str, provider: str) -> tuple[int, int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {
                "id": thread_id,
                "session_id": thread_id,
                "model_provider": provider,
            },
        },
        separators=(",", ":"),
    ).encode("utf-8")
    body = b'{"type":"event_msg","payload":{"msg":"hello"}}\n'
    path.write_bytes(metadata + b"\n" + body)
    token_start = metadata.index(json.dumps(provider).encode("ascii"))
    token_end = token_start + len(json.dumps(provider).encode("ascii"))
    body_start = len(metadata) + 1
    return token_start, token_end, body_start


def make_state_db(path: Path, rows: list[tuple[str, Path, str, str]]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            history_mode TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO threads(id, rollout_path, model_provider, history_mode) VALUES (?, ?, ?, ?)",
        [(thread_id, str(rollout), provider, mode) for thread_id, rollout, provider, mode in rows],
    )
    connection.commit()
    connection.close()


def make_history_db(path: Path, offsets: list[tuple[str, int, int, int]]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE thread_history_projection_state (
            thread_id TEXT PRIMARY KEY,
            next_rollout_byte_offset INTEGER NOT NULL,
            next_rollout_ordinal INTEGER NOT NULL
        );
        CREATE TABLE thread_turns (
            thread_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            rollout_ordinal INTEGER NOT NULL,
            status TEXT NOT NULL,
            rollout_byte_offset INTEGER,
            rollout_end_byte_offset INTEGER,
            PRIMARY KEY (thread_id, turn_id)
        );
        """
    )
    for thread_id, projection, turn_start, turn_end in offsets:
        connection.execute(
            "INSERT INTO thread_history_projection_state VALUES (?, ?, 3)",
            (thread_id, projection),
        )
        connection.execute(
            "INSERT INTO thread_turns VALUES (?, ?, 1, 'completed', ?, ?)",
            (thread_id, f"turn-{thread_id}", turn_start, turn_end),
        )
    connection.commit()
    connection.close()


class CodexProviderRenameTests(unittest.TestCase):
    def test_openai_migration_removes_source_provider_definition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            config = home / "config.toml"
            original = (
                'model_provider = "old"\n'
                "[model_providers.old]\n"
                'name = "Synthetic"\n'
            )
            config.write_text(original, encoding="utf-8")
            backup = home / "backup"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "openai",
                    "--codex-home",
                    str(home),
                    "--keep-model-providers",
                    "--apply",
                    "--backup-dir",
                    str(backup),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('model_provider = "openai"', config.read_text(encoding="utf-8"))
            self.assertNotIn("[model_providers.old]", config.read_text(encoding="utf-8"))

    def test_no_backup_applies_without_backup_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            rollout = home / "sessions/2026/01/01/rollout.jsonl"
            write_rollout(rollout, "thread", "old")
            state = home / "state_1.sqlite"
            make_state_db(state, [("thread", rollout, "old", "legacy")])
            config = home / "config.toml"
            config.write_text('model_provider = "old"\n', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "openai",
                    "--codex-home",
                    str(home),
                    "--apply",
                    "--no-backup",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("without backup", result.stdout)
            self.assertFalse(any(path.name.startswith("codex-provider-rename-backup-") for path in home.iterdir()))
            first = json.loads(rollout.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["payload"]["model_provider"], "openai")
            connection = sqlite3.connect(state)
            provider = connection.execute("SELECT model_provider FROM threads WHERE id = 'thread'").fetchone()[0]
            connection.close()
            self.assertEqual(provider, "openai")

    def test_clean_openai_config_allows_history_migration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            rollout = home / "sessions/2026/01/01/rollout.jsonl"
            write_rollout(rollout, "thread", "old")
            make_state_db(
                home / "state_1.sqlite",
                [("thread", rollout, "old", "legacy")],
            )
            (home / "config.toml").write_text('model_provider = "openai"\n', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "openai",
                    "--codex-home",
                    str(home),
                    "--keep-model-providers",
                    "--apply",
                    "--no-backup",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("built-in OpenAI provider", result.stdout)
            connection = sqlite3.connect(home / "state_1.sqlite")
            provider = connection.execute(
                "SELECT model_provider FROM threads WHERE id = 'thread'"
            ).fetchone()[0]
            connection.close()
            self.assertEqual(provider, "openai")

    def test_fix_null_providers_repairs_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            rollout = home / "sessions/2026/01/01/rollout-null.jsonl"
            rollout.parent.mkdir(parents=True)
            metadata = json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": "thread-null", "model_provider": None},
                },
                separators=(",", ":"),
            ).encode("utf-8")
            body = b'{"type":"event_msg"}\n'
            rollout.write_bytes(metadata + b"\n" + body)
            size = rollout.stat().st_size
            body_start = len(metadata) + 1
            state = home / "state_1.sqlite"
            make_state_db(state, [("thread-null", rollout, "old", "paginated")])
            history = home / "thread_history_1.sqlite"
            make_history_db(history, [("thread-null", size, body_start, size)])
            (home / "config.toml").write_text('model_provider = "old"\n', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "openai",
                    "--codex-home",
                    str(home),
                    "--fix-null-providers",
                    "--apply",
                    "--no-backup",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            first = json.loads(rollout.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["payload"]["model_provider"], "openai")
            connection = sqlite3.connect(history)
            projection = connection.execute(
                "SELECT next_rollout_byte_offset FROM thread_history_projection_state"
            ).fetchone()[0]
            turn = connection.execute(
                "SELECT rollout_byte_offset, rollout_end_byte_offset FROM thread_turns"
            ).fetchone()
            connection.close()
            self.assertEqual(projection, size + 4)
            self.assertEqual(turn, (body_start + 4, size + 4))

    def test_dry_run_is_read_only_and_apply_repairs_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            active = home / "sessions/2026/01/01/rollout-active.jsonl"
            archived = home / "archived_sessions/rollout-archived.jsonl"
            legacy = home / "sessions/2026/01/02/rollout-legacy.jsonl"
            active_info = write_rollout(active, "thread-active", "old")
            archived_info = write_rollout(archived, "thread-archived", "old")
            write_rollout(legacy, "thread-legacy", "old")
            active_size = active.stat().st_size
            archived_size = archived.stat().st_size
            config = home / "config.toml"
            config.write_text(
                'model_provider = "old"\n'
                '[projects."/tmp/example"]\n'
                'model_provider = "old"\n'
                '[model_providers.old]\n'
                'name = "Synthetic"\n'
                'base_url = "https://example.test/v1"\n',
                encoding="utf-8",
            )
            state = home / "state_1.sqlite"
            make_state_db(
                state,
                [
                    ("thread-active", active, "old", "paginated"),
                    ("thread-archived", archived, "old", "paginated"),
                    ("thread-legacy", legacy, "old", "legacy"),
                ],
            )
            history = home / "thread_history_1.sqlite"
            make_history_db(
                history,
                [
                    (
                        "thread-active",
                        active_size,
                        active_info[2],
                        active_size,
                    ),
                    (
                        "thread-archived",
                        archived_size,
                        archived_info[2],
                        archived_size,
                    ),
                ],
            )

            before = {
                path: path.read_bytes()
                for path in (active, archived, legacy, config, state, history)
            }
            dry_run = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "new-provider",
                    "--codex-home",
                    str(home),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
            self.assertIn("DRY RUN (read-only)", dry_run.stdout)
            self.assertEqual(before, {path: path.read_bytes() for path in before})

            backup = home / "backup"
            apply = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "new-provider",
                    "--codex-home",
                    str(home),
                    "--apply",
                    "--backup-dir",
                    str(backup),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(apply.returncode, 0, apply.stdout + apply.stderr)
            self.assertTrue((backup / "manifest.json").exists())

            for path in (active, archived, legacy):
                first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(first["payload"]["model_provider"], "new-provider")

            connection = sqlite3.connect(state)
            self.assertEqual(connection.execute("SELECT DISTINCT model_provider FROM threads").fetchall(), [("new-provider",)])
            connection.close()

            delta = len(json.dumps("new-provider")) - len(json.dumps("old"))
            connection = sqlite3.connect(history)
            self.assertEqual(
                connection.execute("SELECT next_rollout_byte_offset FROM thread_history_projection_state ORDER BY thread_id").fetchall(),
                [(active_size + delta,), (archived_size + delta,)],
            )
            self.assertEqual(
                connection.execute("SELECT rollout_byte_offset, rollout_end_byte_offset FROM thread_turns ORDER BY thread_id").fetchall(),
                [
                    (active_info[2] + delta, active_size + delta),
                    (archived_info[2] + delta, archived_size + delta),
                ],
            )
            connection.close()

            config_text = config.read_text(encoding="utf-8")
            self.assertIn('model_provider = "new-provider"', config_text)
            self.assertIn("[model_providers.new-provider]", config_text)
            self.assertNotIn('[model_providers.old]', config_text)

    def test_compressed_rollout_blocks_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            compressed = home / "sessions/2026/01/01/rollout-compressed.jsonl.zst"
            compressed.parent.mkdir(parents=True)
            compressed.write_bytes(b"not actually zstd")
            (home / "config.toml").write_text('model_provider = "old"\n', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--from",
                    "old",
                    "--to",
                    "new",
                    "--codex-home",
                    str(home),
                    "--apply",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compressed .jsonl.zst", result.stdout + result.stderr)
            self.assertEqual((home / "config.toml").read_text(encoding="utf-8"), 'model_provider = "old"\n')


if __name__ == "__main__":
    unittest.main()
