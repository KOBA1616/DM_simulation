# PaymentPlan 設計メモ

最終更新: 2026-03-16

目的: ランタイムにおけるコスト判定と支払い実行を単一路線化するための `PaymentPlan` 構造体の設計。

ゴール:
- 判定経路（合法手生成）と実行経路が同一の計算結果を共有できるようにする。
- `PASSIVE` と `ACTIVE_PAYMENT` を両方表現できる中間表現を定義する。

設計サマリ:
- `PaymentPlan` は以下を含む（Python プロトタイプに合わせたフィールド）:
  - `base_cost` (int): 元のカードコスト
  - `adjusted_after_passive` (int): PASSIVE 適用後のコスト
  - `total_passive_reduction` (int)
  - `passive_ids` (vector<string>): 適用された PASSIVE 定義の識別子（C++ 側では現状 `name` を一時利用）
  - `active_reduction_id` (optional<string>) / `active_units` (optional<int>) / `active_reduction_amount` (int)
  - `final_cost` (int): 実際に支払うべき最終コスト

互換性ノート:
- 現状 C++ の `CostReductionDef` には Python の `id` 相当フィールドがない。まずは `name` を識別子として利用し、JSON スキーマ/`CostReductionDef` に `id` を追加する移行を次フェーズで行う。

段階的実装プラン:
1. C++ 側で `PaymentPlan` 型を導入（軽量な POD 構造体）。
2. `ManaSystem::get_adjusted_cost` の呼び出しポイントを見直し、`evaluate_cost(card_def, units, active_id)` を呼び出して `PaymentPlan` を得る（読み取り専用）。
3. `CostPaymentSystem` と実行経路（`PLAY_FROM_ZONE`）は `PaymentPlan` を受け取り、そのまま `auto_tap_mana` / `consume_mana` を行う。
4. `CommandDef`/バインディング経路に `payment_*` メタを伝搬して `active_reduction_id` を指定できるようにする（editor/AI → engine）。

注意点:
- `min_mana_cost` の合成ルール（複数 PASSIVE の floor をどのように合成するか）は仕様化が必須。現時点は最大値を floor とする挙動を想定。
- マルチ文明（required_civs）チェックは `PaymentPlan` ではなく支払い検査段階で行うが、`PaymentPlan` に必要文明の要約情報を含めることは有用。

次のタスク:
- `core/card_json_types.hpp::CostReductionDef` に `id` フィールドを追加するための JSON スキーマ変更案を作成
- `ManaSystem` と `CostPaymentSystem` の `evaluate_cost` 呼び出し面を差し替える PR を作成
