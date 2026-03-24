# FIXED -> STAT_SCALED Migration Guide

目的: 既存の `COST_MODIFIER` 定義で `value` (固定減少) を使用しているカードを、後方互換を保ちながら `STAT_SCALED` 仕様に移行する手順を示します。

ステップ:

1. 監査
   - `tools/cards_audit.py` を実行し、`COST_MODIFIER` に `value_mode=STAT_SCALED` が指定されているが `stat_key`/`per_value` が欠落しているカードを特定します。

2. 自動変換（推奨、補助ツール）
   - `tools/data_migration_fix_to_stat_scaled.py` の `migrate_card()` を利用して、1枚ずつカードを変換できます。
   - 変換ルール（簡易）:
     - `value_mode=FIXED` と `value` を `STAT_SCALED` に変換
     - `per_value` = 元の `value`
     - `min_stat` = 1
     - `stat_key` は条件(Filter)に種族があれば `CREATURES_PLAYED`、無ければ `GENERIC_USAGE` を設定（手動で上書き推奨）

3. 手動レビュー
   - 自動変換後、カードごとに `stat_key` と `per_value` がゲーム内文脈に適切かを確認してください。
   - 特に `GENERIC_USAGE` は暫定キーです。実運用では `CardTextResources` に登録された適切な `stat_key` を使用してください。

4. 互換レイヤ（エンジン）
   - `dm_toolkit/payment.py` の `_merged_passive_definitions()` は `value_mode` 未指定を `FIXED` と扱い、`STAT_SCALED` を解釈します。移行中は両方の解釈が混在しても動作するため、カードの段階的移行が可能です。

5. CI と監査
   - 変換後の JSON を `tools/cards_audit.py` で再検査してください。
   - 変換完了後、CI に新たな監査ジョブを追加し、`COST_MODIFIER` が `STAT_SCALED` を使用する場合に `stat_key` と `per_value` が含まれていることを強制することを検討してください。

注意点:
- 自動変換はヒューリスティックです。戦略的設計判断（例: per_value を 1 にするか元の値を使うか）はカードデザイナーが最終決定してください。
- 一時的に `GENERIC_USAGE` のような暫定キーを利用する場合、エンジン側で未定義キーが参照されたときに fail-fast する機構を検討してください。

簡単な実行例 (Python):

```python
from tools.data_migration_fix_to_stat_scaled import migrate_card
import json

cards = json.load(open('data/cards.json'))
new_cards = [migrate_card(c) for c in cards]
json.dump(new_cards, open('data/cards.migrated.json','w'), ensure_ascii=False, indent=2)
```
