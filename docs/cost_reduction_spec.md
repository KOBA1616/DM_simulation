# Cost Reduction（常在・能動）仕様

目的: `PASSIVE`（常在）と `ACTIVE_PAYMENT`（能動）の軽減を統一的に扱うための優先順位と合成ルールを明文化する。

適用対象フィールド（データモデル）
- `cost_reductions[]` エントリ: `id`, `name`, `unit_cost`, `filter`, `min_mana_cost`, `max_units`, `priority`(任意)
- コマンド側: `payment_mode`, `reduction_id`, `payment_units`

基本原則
1. 決定順序
   - Passive（常在） → Active（能動） の順で適用する。
   - Passive 同士は `priority`（数値が小さいほど高優先）でソートして順次適用。`priority` がない場合は定義ファイル中の出現順で適用する。

2. 合成ルール
   - Passive 合成は“累積減算”方式を採用する。
     - 各 `PASSIVE` エントリはその `filter` にマッチするカード/操作に対して、最大 `max_units` まで `unit_cost` を単位として減算する。
     - Passive の適用はベースコスト（カードの `cost` または参照コスト）に対して行い、どの Passive も `min_mana_cost` を下回ることは許容されない（各エントリごとではなく、最終的なコストに対して全体の `min_mana_cost` をチェックする）。
   - Active はプレイヤー選択であり、`payment_units` を使って追加減算を行う。Active 適用時も `min_mana_cost` を下回らないようにクランプする。

3. フィルタ適用と選択性
   - `filter` が指定された Passive は、そのフィルタに一致するカード/操作のみを対象とする。
   - 複数の Passive が同一カードに適用される場合、合成ルール（上記累積減算）に従う。

4. 非負保証と下限
   - 最終的な支払いコストは常に 0 未満にならない。実運用ではさらに `min_mana_cost` による下限を強制する。

5. 後方互換性
   - 既存の `str_val`/`name` ベースの参照は引き続きサポートするが、実行路は `reduction_id`（一意識別子）を最優先で使用する。
   - `reduction_id` が欠落している既存データはロード時に `generate_missing_ids` で補完する。

6. 優先度の衝突検出
   - 同一 `reduction_id` の重複や、複数エントリが競合して不整合を生む場合、保存時バリデーションでエラーまたは警告を出す（設定によりブロッキング/非ブロッキングを選択可能）。

7. テスト行列（簡易）
   - Passive 単体: 1エントリで期待通りにコストが減るか。
   - Passive 複数: 優先度・出現順での合成結果が安定しているか。
   - Passive + Active: 受動減算後に能動減算を行って妥当な結果になるか。
   - Filter 適用: フィルタが機能しているか。
   - min_mana_cost/下限: 適切にクランプされるか。

実装ノート
- エンジン実装は `evaluate_cost(...) -> PaymentPlan` を返す設計とする。`PaymentPlan` は passive_total, active_choice, effective_cost を含む。
- UI/エディタは `payment_preview(units)` API を提供し、即時に試算結果を表示する。

ドキュメント: このファイルは `CARD_EDITOR_REFACTOR_TDD_PLAN.md` の「常在コスト軽減のデータソースを統合する」項と連携します。