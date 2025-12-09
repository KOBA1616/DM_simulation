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

## [2025-03-XX] Phase 6-8: Engine Core & Advanced Mechanics

### Phase 6: 多色・進化・高度な選択 (Multi-color, Evolution, Advanced Selection)

1.  **多色システム (Multi-color System)**
    *   **データ構造**: `CardDefinition.civilizations` を単一から配列へ変更し、複数文明に対応。
    *   **厳格なマナ支払い**: `ManaSystem` にバックトラック法を用いた厳密なコスト支払いロジック (`solve_payment`) を実装し、多色カードが1枚につき1文明のみを提供することを保証。
    *   **タップイン処理**: `TapInUtils` を導入し、多色カードがマナゾーンに置かれる際にタップインされるルール（および `untap_in` キーワードによる例外）を実装。

2.  **進化・階層構造 (Evolution & Hierarchy)**
    *   **階層構造**: `CardInstance` に `underlying_cards` を追加し、進化元などのカードスタックをサポート。
    *   **進化クリーチャー**: `ActionGenerator` を拡張し、進化クリーチャーの使用時に進化元を選択するアクションを生成可能に。
    *   **NEOクリーチャー**: 通常召喚と進化召喚の両方を選択可能にするロジックを追加。
    *   **クリーンアップ処理**: `ZoneUtils::on_leave_battle_zone` を実装し、一番上のカードがバトルゾーンを離れた際、下のカードが墓地に置かれるルールを統一的に適用。

3.  **高度な選択 (Advanced Selection)**
    *   **フィルタリング**: `FilterDef` に `selection_mode` (MIN/MAX/RANDOM) と `selection_sort_key` (COST/POWER) を追加。
    *   **ロジック**: `PendingEffectStrategy` にて、選択候補をソートおよびフィルタリングし、「パワーが最小のクリーチャーを選ぶ」等の処理を自動化。
    *   **複合条件**: `FilterDef` に `and_conditions` を追加し、再帰的なAND条件フィルタをサポート。
    *   **逆選択 (Inverse Selection)**: `ActionDef` に `inverse_target` フラグを追加。「選ばなかったものを破壊」するロジックを `DestroyHandler` 等に実装。

### Phase 7: 複雑な領域移動・パッシブ効果 (Allocation & Passives)

4.  **複雑な領域移動 (Complex Allocation)**
    *   **共通基盤**: `ZoneUtils::find_and_remove` を実装し、バトルゾーン、手札、マナ、シールド、墓地、および **Effect Buffer** から安全にカードを検索・移動させるロジックを共通化。
    *   **ハンドラの統合**: `DestroyHandler`, `ReturnToHandHandler`, `ManaChargeHandler` が `ZoneUtils` を使用するように改修し、一時領域 (Buffer) からの移動をサポート。「3枚見て、1枚手札、1枚マナ、1枚墓地」のような振り分け処理が可能に。
    *   **カードの下に置く**: `EffectActionType::MOVE_TO_UNDER_CARD` およびハンドラを実装。

5.  **パッシブ効果・情報開示 (Passive Effects & Reveal)**
    *   **パッシブ効果システム**: `PassiveEffectSystem` を実装。`EffectResolver::get_creature_power` をフックし、常在型能力（全体パワー修正など）を動的に適用。
    *   **公開アクション**: `EffectActionType::REVEAL_CARDS` およびハンドラを実装。

### Phase 8: スタックシステムとビルド安定化 (Stack System & Build Stabilization)

6.  **トリガースタック (Trigger Stack System) - 実装済み**
    *   トリガー能力（CIP等）を即時実行せず、`PendingEffect` (Type: `TRIGGER_ABILITY`) として待機リストに追加。
    *   `EffectResolver` にて `RESOLVE_EFFECT` アクションを介して順次実行する仕組みを実装。
    *   `python/tests/test_trigger_stack.py` にて基本動作（トリガーの保留と解決）を検証済み。

7.  **ビルド環境の修復と安定化 (Build Fixes)**
    *   `CardDefinition` の文明定義（単数→複数）変更に伴うコンパイルエラー（`ManaSystem`, `JsonLoader`, `CsvLoader`, `TensorConverter`）を修正。
    *   Pythonバインディング (`bindings.cpp`) のインクルード漏れや型定義の不整合（`CardKeywords`, `Zone` Enum）を修正し、`dm_ai_module` の正常ビルドを回復。

8.  **超魂クロス (Metamorph) および 多色マナ厳格化 (Strict Multicolor Mana) - 実装済み**
    *   **超魂クロス**: 817.1a (通常能力との共存) および 817.1b (下のカードから超魂能力のみ付与) のルールを実装。`GenericCardSystem::resolve_trigger` にて、`CardDefinition.metamorph_abilities` を用いて能力を適切に合成するロジックを確立。
    *   **多色マナ支払いの厳格化**: `ManaSystem` にバックトラック法を用いた厳密なコスト支払いロジック (`solve_payment`) を実装。多色カードが1枚につき1文明のみを提供し、かつ必要な全文明を過不足なく満たす組み合わせを探索・適用するように改修。
