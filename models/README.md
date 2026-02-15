モデルフォルダ管理

- スクリプト: `scripts/cleanup_models.py`
- 設定: `models/cleanup_config.json`

概要:
- 古いモデルファイルを `models/archive/` に移動するか削除するユーティリティ
- デフォルトは dry-run 有効。まず `--dry-run` で確認してください。

基本の実行例:
```
python scripts/cleanup_models.py --models-dir models --keep 3 --dry-run
```

運用案:
- 定期実行: Windows では Task Scheduler、CI では GitHub Actions を利用して週次実行
- アーカイブ先は `models/archive/` です。容量を確保したら古いアーカイブを別ストレージへ移動
- 自動削除する場合は `--delete --confirm` を使用するが不可逆なので注意
