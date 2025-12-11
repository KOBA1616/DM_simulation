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
