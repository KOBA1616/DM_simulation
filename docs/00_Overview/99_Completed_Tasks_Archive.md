# 完了したタスクのアーカイブ (Completed Tasks Archive)

このドキュメントは `00_Status_and_Requirements_Summary.md` から移動された、完了済み要件の履歴です。

---

## Phase 0.5-1.0: Foundation & Core Features (Archived)

### Engine & AI Core
*   **MCTS & AlphaZero**: 基礎実装完了。
*   **並列実行 (`ParallelRunner`)**: 実装完了。
*   **Pythonバインディング**: `dm_ai_module` として実装完了。

---

## Phase 1.5: カードエディタ UI/UX 改善 (Completed)

カードデータの入力効率と表現力を向上させ、複雑なカード（多色、ツインパクトなど）の実装を可能にするための改修。

1.  **多色（マルチカラー）カード実装支援**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `CivilizationSelector` ウィジェットを作成し、`CardEditForm` に統合。
        *   単一選択ではなく、複数の文明（Fire, Water, etc.）をチェックボックス形式で選択可能にし、`civilizations` リストとしてJSONに保存する機能を実装。

2.  **ツインパクトカードの視覚的統合**
    *   **完了日**: 2024/05/25
    *   **実装内容**:
        *   `CardEditForm` に「Is Twinpact?」チェックボックスを追加。
        *   チェック時に呪文側編集フォーム（`SpellSideWidget`）を動的に表示するUIロジックを実装。
        *   JSONデータの `spell_side` フィールドへのネスト保存に対応。

3.  **キーワード能力とトリガー設定の整理**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `EffectEditForm` の `TriggerType` プルダウンから重複していた `S_TRIGGER` を削除。
        *   `CardEditForm` のキーワード能力チェックボックス群（`shield_trigger`, `blocker` 等）を集約し、設定フローを統一。

4.  **アクションタイプ「数字を宣言(選択)」の追加**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `ACTION_UI_CONFIG` に `DECLARE_NUMBER` (内部的には `SELECT_NUMBER`) を追加。
        *   ガチンコ・ジャッジや特定コスト指定などのために、プレイヤーが数値入力を行うUIをサポート。

5.  **未反映アクション・条件・効果のGUI反映**
    *   **完了日**: 2024/05/17
    *   **実装内容**:
        *   `RETURN_TO_HAND` (バウンス), `SEARCH_DECK_BOTTOM`, `REVOLUTION_CHANGE` (革命チェンジ), `FRIEND_BURST`, `SELECT_NUMBER` 等のエンジン機能をエディタの選択肢に追加。
        *   `MANA_ARMED` (マナ武装), `CIVILIZATION_MATCH` 等の条件判定ロジックをエディタで設定可能にした。

6.  **カードテキスト自動生成機能**
    *   **完了日**: 2024/05/26
    *   **実装内容**:
        *   `CardTextGenerator` クラスを実装。入力されたJSONデータ（効果、対象、数値）に基づき、自然言語（日本語）のカードテキストをリアルタイムでプレビュー生成する機能を追加。

---

## Phase 1.6: エンジン機能拡張 (Completed)

エディタで作成されたデータを正しく処理するためのエンジン側対応。

1.  **ツインパクトカードのスキーマ対応**
    *   **完了日**: 2024/05/25
    *   **実装内容**:
        *   `CardDefinition` および `CardData` 構造体に `spell_side` (std::shared_ptr) フィールドを追加。
        *   `JsonLoader` による再帰的な読み込みと、Pythonバインディングを通じたアクセスを実装。

2.  **高度なメカニクス実装**
    *   **完了日**: 2024/05/30
    *   **実装内容**:
        *   **Hyper Energy**: `HyperEnergyHandler` およびコスト軽減ロジックの実装。
        *   **Revolution Change**: `RevolutionChangeHandler` および攻撃時の宣言ウィンドウ、入れ替えロジックの実装。
        *   **Just Diver**: `CardKeywords` へのフラグ追加と、選ばれない効果（`TargetUtils`）の期間制御の実装。

3.  **アクション/効果ハンドラの拡充**
    *   **完了日**: 2024/05/30
    *   **実装内容**:
        *   `ShieldBurn` (シールド焼却), `GrantKeyword` (能力付与), `SearchDeckBottom`, `ReturnToHand` 等の専用ハンドラを `src/engine/systems/card/handlers` に実装し、`GenericCardSystem` から委譲する構成を確立。

---

## Phase 2 & 3: AI Evolution & Refactoring (Completed)

AI知能の進化、不完全情報対応、およびシステム基盤の安定化に関連する完了済みタスク。

### 1. Refactoring & Stabilization (Phase 3.0)
*   **テストディレクトリ再編 (Test Directory Reorganization)**
    *   `src/main_test.cpp` を `tests/cpp/` に移動し、`tests/python/` との役割分担を明確化しました。
*   **CardInstance構造改善 (CardInstance Structure Improvement)**
    *   `src/core/game_state.hpp` の `CardInstance` 構造体に `owner` (PlayerID) フィールドを追加し、Pythonバインディングからもアクセス可能にしました。
*   **GUI安定化 (GUI Stabilization)**
    *   `MCTS` クラスの Python バインディングに `search` メソッドを露出し、GUI からの呼び出しエラーを修正しました。
*   **ParallelRunner メモリリーク (ParallelRunner Memory Leak)**
    *   `collect_data` フラグを導入し、検証実行時に膨大なゲーム状態を保持しないように修正しました。
*   **PhaseManagerバインディング修正 (PhaseManager Binding Fix)**
    *   `dm_ai_module.PhaseManager` に `check_game_over` メソッドをバインディングし、`AttributeError` を解消しました。
*   **新規能力実装 (New Ability Implementation)**
    *   「数字を宣言し、そのコストの呪文を唱えられなくする」ロック効果 (`LOCK_SPELL_BY_COST`) をエンジンに実装しました。
*   **リーサルソルバー改善 (Lethal Solver Improvement)**
    *   除去、タップ、ブロック不可などを考慮した高度な判定ロジックを実装しました。
    *   シールドブレイク優先順序をシミュレートするグリーディヒューリスティックを採用しました。

### 2. Imperfect Information (Phase 3.1)
*   **相手手札推定器 (Opponent Hand Estimator)**
    *   `dm_toolkit/ai/inference` に実装済み。`DeckClassifier` と `HandEstimator` を実装。
*   **PIMC (Perfect Information Monte Carlo) サーチ**
    *   `src/ai/inference/pimc_generator.cpp` にて候補プールからのサンプリング生成機能を実装。
    *   `src/ai/self_play/parallel_runner.cpp` に `run_pimc_search` を実装し、並列PIMC探索を実現。

### 3. Self-Evolving Ecosystem (Phase 3.2)
*   **自動学習サイクル (The Automation Loop)**
    *   `dm_toolkit/training/automation_loop.py` にて、Collector, Trainer, Gatekeeper の完全自動ループを実装済み。
*   **世代管理とストレージ戦略 (Generation Management)**
    *   `dm_toolkit/training/generation_manager.py` にて、階層化保存（Production/Population/Hall of Fame）とデータ自動削除を実装済み。
*   **デッキ進化 (Deck Evolution)**
    *   `DeckEvolution` クラスおよび `calculate_interaction_score` ロジックを `src/ai/evolution/` (C++) に移植し、高速化を実現しました。

### 4. User Requested Enhancements (Phase 3.5)
*   **カードエディタ UI/UX 改善 (Card Editor UI/UX)**
    *   3ペイン構成への刷新、簡易プレビュー機能、文明選択UIの装飾、呪文パワー入力マスク等を実装済み。
*   **MCTS 可視化機能の復旧 (Restore MCTS Visualization)**
    *   `MCTS` クラスに `get_last_root` メソッドを追加・公開し、GUIでのツリー可視化を可能にしました。
*   **テストカバレッジの拡充 (Test Coverage)**
    *   エンジン実装済みの「呪文ロック効果」について、エンドツーエンドのテストケースを作成し動作検証を行いました。

---

## Phase 4.0: Refactoring & Stabilization Updates (Archived from Doc 00)

### 1.1 Core Engine Enhancements
*   **Civilization Logic Refactoring**: `Civilization` logic was refactored to support multi-civilization cards using `std::vector` in C++.
*   **Cost Reduction System**: Implemented a robust cost reduction system (Base -> Passive -> Active -> G-Zero) and verified via `tests/test_cost_modifier.py`.
*   **Loop Detection**: Implemented `GameState::calculate_hash` and history tracking to detect infinite loops.
*   **Zone Refactoring**: Refactored `Hand`, `Mana`, `Grave` actions into a unified `MOVE_CARD` action type.
*   **Atomic Actions**: Consolidated atomic actions (Draw, Tap, Break Shield) and verified them via `tests/test_atomic_actions.py`.

### 1.2 Card Editor Ver 2.1 Enhancements (GUI)
*   **Modular Architecture**: Re-implemented the editor using a modular design (`CardEditForm`, `LogicTreeWidget`, `DataManager`).
*   **Filter Editor**: Implemented a comprehensive UI for editing `FilterDef` (Zones, Civs, Races, Power/Cost ranges).
*   **Variable Linking**: Implemented "Smart Link" and `VariableLinkWidget` to connect action outputs to subsequent inputs (e.g., "Count Creatures" -> "Draw X").
*   **Localized UI**: Extensive Japanese localization for menus, labels, and enums (`python/gui/localization.py`).
*   **Visual Enhancements**: Added color coding for civilizations and improved layout for lists.

### 1.3 Critical Fixes
*   **Static Library Duplication**: Resolved singleton duplication by building `dm_core` as an OBJECT library in CMake.
*   **Process Crash**: Fixed `gilstate_tss_set` error during thread termination in `ParallelRunner`.
*   **Shift-JIS Support**: Verified source encoding compliance.

---

## Phase 4.1: Recent Enhancements & Fixes (Archived)

`00_Status_and_Requirements_Summary.md` より移動された、直近の実装済み機能および修正事項。

### 1. Engine & Logic Expansions
*   **呪文を唱えるアクション (CAST_SPELL)**: `CAST_SPELL` アクションを追加し、手札等からコスト踏み倒しで唱える処理を実装済み。
*   **クリーチャーを出すアクション (PUT_CREATURE)**: `PUT_CREATURE` アクションを追加し、バトルゾーンへの直接配置（召喚酔いあり、CIP誘発）を実装済み。
*   **キーワード能力付与 (GRANT_KEYWORD)**: `GRANT_KEYWORD` アクションを追加し、キーワードと持続ターン数を設定可能にしました。
*   **汎用カード移動 (MOVE_CARD)**: `MOVE_CARD` アクションをGUIおよびエンジンに追加し、移動先ゾーン（Destination Zone）を選択可能にしました。

### 2. Scenario Editor Improvements
*   **ゾーン管理の構造化**: タブ管理（`QTabWidget`）を導入し、General設定と各ゾーンを分離しました。
*   **カード検索・追加機能**: カード検索ウィジェット (`CardSearchWidget`) を実装し、フィルタリングを可能にしました。
*   **ドラッグ＆ドロップ**: 検索結果からのドラッグ＆ドロップによるカード追加に対応しました。

### 3. Card Editor GUI Improvements
*   **多色表示の改善**: プレビュー表示にて、多色カードの背景をグラデーション（または混合色）で表示するように対応しました。
*   **レイアウト調整**: `CardPreviewWidget` のレイアウト（マナコスト位置、パワー位置、種族表示など）を要求通りに再構築しました。
*   **ツインパクト対応**: ツインパクト用のスプリットビュー（上下分割レイアウト）を追加し、各パートの表示を実装しました。

### 4. Other Improvements & Fixes
*   **Game Info ウィンドウの整理**: `GameWindow` のレイアウトを再設計し、操作系とAI設定系を分離しました。
*   **重要バグ修正 (Card Stats)**: `test_card_stats_win_contribution` の無限ループ問題をC++エンジン側で修正しました。

---

## Phase 4.2: User Requested Enhancements (Completed)

`00_Status_and_Requirements_Summary.md` の Section 3.1 より移動された、ユーザー要望およびGUI/ロジック改善タスク。

### 1. Engine & Logic Extensions
*   **フレンド・バースト (Friend Burst)**: `CardKeywords` へのフラグ追加と `FriendBurstHandler` によるタップ・呪文詠唱ロジックを実装済み。
*   **呪文ロック (Lock Spell)**: `LOCK_SPELL_BY_COST` を汎用化し、`ModifierHandler` に統合。
*   **革命チェンジ簡易実装 (Revolution Change)**: 編集フォームでの条件設定（種族・コスト）を実装し、ロジックツリーへの自動生成機能（`ON_ATTACK_FROM_HAND` + `REVOLUTION_CHANGE` アクション）を追加しました。
*   **条件付きS・トリガー (Conditional S-Trigger)**: エンジン側での条件判定ロジックを汎用化しました。
*   **トリガー・リアクション拡張**: ストライク・バックの実装、攻撃時リアクションの条件再確認、攻撃中断時の制約追加。
*   **原子アクションの汎用化**: `ActionDef` への `condition` 追加、および `Draw/Destroy/Shield` ハンドラの変数リンクバグ修正。
*   **コスト軽減の汎用化**: `APPLY_MODIFIER` (Mode: COST) アクションと変数リンクによるコスト軽減ロジックを実装。

### 2. GUI & Visualization Improvements
*   **日本語化の推進**:
    *   `CardTextGenerator` によるアクションテキストの自然な日本語生成。
    *   `localization.py` の大幅拡充（アクションタイプ、トリガー、ゾーン名、UIラベル等）。
*   **カードプレビュー改善**:
    *   ツインパクトカードのパワー表示位置（左下）、黒枠線、マナコスト黒字化等のスタイル調整。
    *   多色カードのグラデーション背景対応。
*   **カードエディタ改善 (Card Editor)**:
    *   **ID管理隠蔽**: 編集フォームからIDフィールドを非表示化。
    *   **ツインパクトロジック統合**: ツインパクト選択時にロジックツリーに「呪文側 (Spell Side)」ノードを自動生成し、呪文効果をツリー構造で管理可能にしました。
    *   **キーワード整理**: 進化 (Evolution)、メタカウンター、アンタップイン等のキーワードをチェックボックスから削除し、カードタイプ連動やアクション自動生成へ移行しました。
    *   **革命チェンジ自動生成**: ボタン一つで革命チェンジ用のトリガーとアクションノードを生成する機能を追加しました。
    *   **Reaction Ability UI**: 手札誘発（ニンジャ・ストライク等）専用の編集ウィジェットを実装。
*   **デッキ表示 (ZoneWidget)**:
    *   ゲームボード上のデッキ表示を、全カードを展開するのではなく「デッキ (N枚)」という単一の束として表示するように変更しました。
*   **シナリオエディタ改善**:
    *   使用デッキ指定機能（JSONロード）と、デッキ内検索・配置機能の実装。

### 3. System Improvements
*   **効果バッファの分離**: `effect_buffer` を `GameState` から `Player` 構造体へ移動し、プレイヤーごとの独立管理を実現（バグ防止）。

### 5. GUI Enhancements & User Requests (Phase 4.3)

`00_Status_and_Requirements_Summary.md` の Section 3.1 より移動された、GUI機能拡張タスク。

*   **ロジックマスク (Logic Mask)**
    *   **アクション編集**: 選択したアクションタイプに応じて、不要なフィルタ設定項目（ゾーン、文明など）を自動的に非表示にする機能を `FilterEditorWidget` と `ActionEditForm` に実装しました。
    *   **条件編集**: 選択した条件タイプに応じて、不要な値設定フィールドをマスクする機能を `EffectEditForm` に実装しました。
*   **トリガータイミング包含 (Trigger Timing Inclusion)**
    *   `ConditionDef` に `filter` フィールドを追加し、エンジンのデータ構造を拡張しました。
    *   `EffectEditForm` の条件設定に `FilterEditorWidget` を組み込み、`EVENT_FILTER_MATCH` 条件タイプを選択することで、トリガーイベントに対する詳細なフィルタリング（例：「相手クリーチャーが出た時」）をGUI上で設定可能にしました。
