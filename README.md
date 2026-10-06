# codex-provider-transcript-migrate

Dry-run by default:

```powershell
python codex_provider_rename.py --from old-provider --to new-provider
```

To switch assignments to a built-in provider without renaming custom
`[model_providers.*]` definitions, add:

```powershell
--keep-model-providers
```

If migrating to `openai` after manually removing its old custom provider, a
clean config is allowed: the absent source reference is reported as a warning
instead of blocking history migration.

To also replace old `model_provider:null` rollout metadata with the target
provider, add:

```powershell
--fix-null-providers
```

Apply normally creates a backup. If storage is constrained, `--no-backup`
applies without one and disables automatic rollback.
