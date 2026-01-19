# Backups

このフォルダは古いバージョンや`.bak`ファイルの保管場所です。

現時点のバックアップファイル:

- [CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md.bak](CAST_SPELL_REPLACE_CARD_MOVE_IMPLEMENTATION.md.bak)
- [IF_CONDITION_LABELS.md.bak](IF_CONDITION_LABELS.md.bak)
- [MIGRATION_GUIDE_ACTION_TO_COMMAND.md.bak](MIGRATION_GUIDE_ACTION_TO_COMMAND.md.bak)

復元手順の例:

- 個別ファイルを元に戻す（PowerShell）:

	```powershell
	Move-Item docs\backups\IF_CONDITION_LABELS.md.bak docs\IF_CONDITION_LABELS.md -Force
	```

- Git 管理下で元に戻す（tracked ファイルを HEAD に復元）:

	```bash
	git restore --worktree --source=HEAD -- docs/IF_CONDITION_LABELS.md
	```

古いバックアップが不要であればアーカイブ（zip）して保存するか削除してください。
