# 完成したタスクのアーカイブ (Completed Tasks Archive)

このドキュメントは `00_Status_and_Requirements_Summary.md` から完了したタスクを移動して記録するためのアーカイブです。

## 完了済み (Completed)

### 3.0 [Priority: High] Phase 6: GameCommand アーキテクチャとエンジン刷新 (Engine Overhaul)

1.  **イベント駆動型トリガーシステムの実装**
    *   ハードコードされたフックポイントを廃止し、`TriggerManager` による一元管理へ移行。
    *   **Status**: `TriggerManager`, `GameEvent` クラスの実装とPythonバインディングが完了 (Phase 6.1 Completed)。
2.  **GameCommand (Primitives) の実装**
    *   全てのアクションを `TRANSITION`, `MUTATE`, `FLOW`, `QUERY`, `DECIDE` に分解・再実装。
    *   **Status**: 基本5命令のクラス実装、Pythonバインディング、および `GameState` への統合が完了。Unit Test (`tests/test_game_command.py`) を復元・実装し動作確認済み (Phase 6.2 Completed)。
3.  **アクション汎用化**
    *   **Status**: `MOVE_CARD`、`TAP`、`UNTAP`、`APPLY_MODIFIER`、`MODIFY_POWER`、`BREAK_SHIELD`、`DESTROY_CARD`、`PLAY_CARD`、および `ATTACK` (AttackHandler) のハンドラを `GameCommand` を使用するように移行完了。`GameCommand` の `Zone` に `STACK`, `BUFFER` を追加し、拡張完了 (Phase 6.3 Completed)。

### 2.4 実装上の不整合の修正 (Inconsistencies Resolved)

1.  **C++ コンパイル警告 (ConditionDef) の修正**
    *   `CostHandler`, `ShieldHandler`, `SearchHandler`, `DestroyHandler`, `UntapHandler` 等において、`ConditionDef` のブレース初期化リストを修正し、`missing initializer` 警告を解消しました。
    *   未使用のパラメータ (`unused parameter`) についても修正を行い、ビルドログをクリーンにしました。

2.  **Atomic Action テストの修正**
    *   `tests/python/test_new_actions.py` 内の `test_cast_spell_action` および `test_put_creature_action` を修正しました。
    *   `GenericCardSystem.resolve_effect_with_targets` を呼び出す際、明示的に `CardType` (SPELL/CREATURE) を設定した `card_db` を渡すことで、エンジンが正しい解決パス（呪文は墓地へ、クリーチャーはバトルゾーンへ）を選択するようにしました。

3.  **革命チェンジのデータ構造不整合の修正**
    *   `CardEditor` に「革命チェンジ」チェックボックスを追加し、Engineが期待するルートレベルの `revolution_change_condition` (FilterDef) を生成するように修正しました。
    *   チェックボックス操作に連動して、ロジックツリー内に `Trigger: ON_ATTACK_FROM_HAND` および `Action: REVOLUTION_CHANGE` を自動生成するロジックを実装しました。

4.  **文明指定のキー不整合の修正**
    *   `CardEditor` (Data Manager) が新規カード作成時にリスト形式の `"civilizations"` を使用するように統一しました。
    *   `CardEditForm` は既にリスト形式に対応していましたが、データの保存・読み込み時のレガシーサポート（`civilization` 文字列）は `JsonLoader` (Engine) 側で引き続き担保されます。

5.  **Card Editor UI の改善 (Polish)**
    *   **カードプレビュー**:
        *   ツインパクトカードのパワー表記を「カード全体の左下」に配置しました。
        *   マナコストの円形背景を文明色（多色の場合は等分割グラデーション）で描画するように修正しました。
        *   マナコストの文字色を「黒縁の白文字」（実装上は太字の白文字＋黒い円形枠線）に統一しました。
        *   カードの外枠（選択時の強調部分）を「すべての文明で黒の細線」に統一しました。
    *   **テキスト生成**: `CardTextGenerator` に `EX Life` (EXライフ) のキーワード対応を追加しました。

6.  **汎用コストおよび支払いシステム (General Cost and Payment System) - Step 1: Infrastructure**
    *   **C++ Core Types**: `CostType`, `ReductionType`, `CostDef`, `CostReductionDef` を `src/core/card_json_types.hpp` に定義しました。
    *   **Card Definition Update**: `CardDefinition` に `cost_reductions` フィールドを追加しました。
    *   **Logic Implementation**: `src/engine/cost_payment_system.cpp` を実装し、能動的コスト軽減（Active Cost Reduction）の計算ロジック (`calculate_max_units`) と支払い判定ロジック (`can_pay_cost`) を実装しました。
    *   **Python Binding**: 新しい型とシステムクラスを `dm_ai_module` に公開し、`test_cost_payment_structs.py` による検証を完了しました。

7.  **多色マナ支払いの厳密化 (Strict Multicolor Mana Payment)**
    *   **ManaSystem**: `get_usable_mana_count` に `card_db` 引数を追加し、`solve_payment_internal` を用いた厳密な文明チェック（必要文明を持つタップされていないカードの組み合わせが存在するか）を実装しました。
    *   **EffectResolver**: `PAY_COST` アクション処理時に `auto_tap_mana` の返り値を検証し、支払いに失敗した場合（例：マナ不足や文明不一致）はカードを手札に戻すフォールバック処理を追加しました。
    *   **PhaseStrategies**: 能動的コスト軽減（ハイパーエナジー等）の適用判定においても、厳密なマナチェックが行われるように `get_usable_mana_count` の呼び出しを更新しました。
