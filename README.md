# codex-provider-transcript-migrate

Dry-run by default:

```powershell
python codex_provider_rename.py --from old-provider --to new-provider
```

To leave provider definitions untouched when migrating to a non-OpenAI provider,
add:

```powershell
--keep-model-providers
```

When migrating to `openai`, the script removes the source provider's custom
`[model_providers.*]` tables so they cannot shadow the built-in provider.

To also replace old `model_provider:null` rollout metadata with the target
provider, add:

```powershell
--fix-null-providers
```

Apply normally creates a backup. Rollout backups use hard links when the backup
is on the same volume, so they do not duplicate transcript data. SQLite backups
are separate copies. If storage is constrained, `--no-backup` skips backups and
disables automatic rollback. Rewriting a rollout still stages one full temporary
copy at a time; the dry run reports the largest matching file, which is the
approximate peak rollout staging space. With `--no-backup`, a replacement that
fits in the existing provider field is instead written in place with padding,
so offsets stay unchanged and no full-file copy is needed. A process or power
failure during that small write can damage that rollout's metadata line; longer
provider values still use atomic staged replacement. The script does not run
`VACUUM`: shrinking a SQLite database requires extra temporary space and is
unrelated to changing provider IDs.
