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

## [2025-XX-XX] Phase 5-6: System Verification & Robustness

### 5. テストコードの整理と動作確認
*   `tests/` ディレクトリ内のアドホックなテストを `python/tests/` へ統合。
*   原子アクション (Draw, Mana Charge, Tap, Break Shield, Move Card) の動作を検証する `python/tests/test_atomic_actions.py` の作成と維持。
*   **Status**: 完了 (2025/XX/XX)

### 6. GUIの日本語化拡充
*   `python/gui/card_editor.py` および `localization.py` を更新し、英語のハードコードを排除して日本語表示に対応する。
*   **Status**: 完了 (2025/XX/XX)

### 7. PythonコードのC++移行 (Migration Candidates)
*   パフォーマンス向上とロジックの堅牢化のため、以下のPython実装部分をC++へ移行済み。
    *   **シナリオ実行**: `ScenarioExecutor` (C++) を実装。
    *   **デッキ進化/検証**: `ParallelRunner` (C++) での対戦実行。
    *   **AI学習データ生成の制御**: `DataCollector` の制御をC++へ移管。
*   **Status**: 完了 (2025/XX/XX)

### 8. 変数連携システム (Variable Linking System)
*   アクション間で値を渡すための「実行コンテキスト」を強化し、動的な数値参照を実現。
*   `COUNT_CARDS`, `GET_GAME_STAT`, `SEND_TO_DECK_BOTTOM` などの新規アクションを追加。
*   **Status**: 完了 (2025/XX/XX)

### 9. カードエディタ GUI実装仕様 (Ver 2.0)
*   **標準IDEレイアウト**: 左：ツリービュー / 右：プロパティインスペクタ。
*   **変数リンクシステム**: 安全な変数参照のためのドロップダウンリストによる選択方式。
*   **Status**: 完了 (2025/XX/XX)

### 10. C++移行における改修詳細
*   `DataCollector`, `ParallelRunner` (Scenario), `ParallelRunner` (Deck Evolution) の実装と最適化完了。
*   **Status**: 完了 (2025/XX/XX)
