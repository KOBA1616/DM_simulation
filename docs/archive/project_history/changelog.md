# Completed: Docs reorganization (2026-01-20)

このファイルは 2026-01-20 に実施されたドキュメント整理の完了項目をまとめた記録です。

完了項目:

- 作成: `docs/backups/` ディレクトリ（既存の `.bak` ファイルを集約）
- 作成: `docs/migration/` ディレクトリ（移行ガイド保管）
- 作成: `docs/reference/` ディレクトリ（SPELL 系ドキュメントなど）
- 作成: `docs/guides/` ディレクトリ（実装ガイド類）
- 移動: `MIGRATION_GUIDE_ACTION_TO_COMMAND.md` → `docs/migration/`
- 移動: `IF_CONDITION_LABELS.md`, `CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md` → `docs/guides/`
- 移動: `SPELL_*.md` → `docs/reference/`
- 移動: `*.bak` → `docs/backups/`
- 追加: 各サブフォルダに `README.md` を作成し、短い要約と主要見出しを記載
- 追加: トップ `docs/README.md` を更新し、新しいサブフォルダへのリンクを明示

変更日時: 2026-01-20

差分確認方法:

```bash
git status -- docs
git diff HEAD -- docs
```

元に戻す（例）:

PowerShell:
```powershell
# IF_CONDITION_LABELS.md を元の場所へ戻す例
Move-Item docs\guides\IF_CONDITION_LABELS.md docs\IF_CONDITION_LABELS.md -Force
```

GitでtrackedファイルをHEADに復元する例:
```bash
git restore --worktree --source=HEAD -- docs/IF_CONDITION_LABELS.md
```

備考:
- 追加の自動目次生成や見出し抽出は別タスクとして実施可能です。
