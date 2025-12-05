# 完了済みタスクアーカイブ (Completed Tasks Archive)

## [2024-05-xx] Phase 4-5: Mechanics & GUI Extensions Completed

### 1. Just Diver (ジャストダイバー)
*   **概要:** 「このクリーチャーが出た時、次の自分のターンのはじめまで、相手に選ばれず、攻撃されない」能力の実装。
*   **実装:**
    *   `CardKeywords` に `just_diver` フラグ追加。
    *   `CardInstance` に `turn_played` を追加し、経過ターン判定に使用。
    *   `TargetUtils` および `ActionGenerator` で相手からの選択・攻撃を制限。
*   **検証:** `tests/test_just_diver.py` にて、対象選択不可および期間終了後の解除を確認済み。

### 2. GUI Card Editor Extensions (カードエディタ拡張)
*   **概要:** カードエディタの機能拡張によるデータ作成効率化。
*   **実装:**
    *   **フィルタ詳細化:** 文明、種族、コスト・パワー範囲、状態フラグ（タップ、ブロッカー、進化）のGUI追加。
    *   **トリガー条件 (Condition):** マナ武装、シールド枚数などの条件設定UI追加。
    *   **日本語化:** 各種列挙型、ラベルの日本語化完了。
    *   **コスト軽減UI:** `APPLY_MODIFIER` 選択時の動的ラベル変更（軽減量、期間）。
*   **検証:** `verify_condition_editor_logic.py` 等により、JSONデータの整合性を確認済み。

### 3. Revolution Change (革命チェンジ)
*   **概要:** 攻撃時の入れ替わりギミックの実装。
*   **実装:**
    *   `ON_ATTACK_FROM_HAND` トリガーの実装。
    *   `USE_ABILITY` アクションによる入れ替わり処理。
    *   JSONデータへの `revolution_change_condition` 追加とエディタ対応。
*   **検証:** `tests/test_revolution_change.py` にて動作確認済み。

### 4. Hyper Energy (ハイパーエナジー)
*   **概要:** クリーチャーをタップしてコストを軽減する召喚方法。
*   **実装:**
    *   `COST_REFERENCE` アクションと `GenericCardSystem` によるタップ処理。
    *   コスト計算ロジックへの統合。
*   **検証:** `tests/test_hyper_energy.py` にて動作確認済み。
