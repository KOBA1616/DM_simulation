# cards.json データ移行手順書

作成日: 2026-03-19

目的: `data/cards.json` のスキーマや統計キーを変更した際に、安全にデータを移行・検証する手順を示す。

前提:
- リポジトリのルートに `data/cards.json` があること
- 移行作業はブランチ上で行い、差分を PR で提出すること

手順:

1) 変更点の洗い出し
   - 変更する統計キー名・命名規則をドキュメントに記載する。
   - 例: `DESTROY_COUNT_THIS_TURN` を導入、既存の `OLD_DESTROY_COUNT` を廃止。

2) 互換マップの作成
   - `tools/migrations/` 下に移行スクリプト（Python）を作成し、旧キー→新キーのマッピングを定義する。

3) 自動変換スクリプトの実行（ローカル）
   - 例:

```bash
python tools/migrations/migrate_cards_keys.py --input data/cards.json --output data/cards.migrated.json
```

4) 差分検証（diff）
   - 生成物と元ファイルの差分を確認する。必須チェック項目:
     - 期待するキーが正しく置換されていること
     - その他のフィールドに意図しない変更がないこと
   - コマンド例:

```bash
git --no-pager diff --no-index -U3 data/cards.json data/cards.migrated.json | sed -n '1,200p'
```

5) 契約テストの実行
   - `pytest tests/test_data_stat_key_audit.py` を実行して、移行後ファイルに未知キーが残っていないことを確認する。

6) PR 作成とレビュー
   - 生成済みの `data/cards.migrated.json` を `.migrated` として添付し、PR で差分レビューを依頼する。

差分検証のチェックリスト（回帰防止）:
- すべての `QUERY` の `str_param` が `CardTextResources` または `pipeline_executor` の既知キーに含まれること
- tests がローカルで全て通ること
- 移行スクリプトは idempotent（同じ変換を複数回行っても結果が変わらない）であること

運用メモ:
- CI に移行検証ジョブを追加すると安全（例: PR 時に `tools/stat_key_audit.py` を実行）。
