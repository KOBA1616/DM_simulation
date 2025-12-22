# Status and Requirements Summary (要件定義書 00)

このドキュメントはプロジェクトの現在のステータス、実装済み機能、および次のステップの要件をまとめたマスタードキュメントです。

本要件定義書はUTF-8で記述することを前提とします。
また、GUI上で表示する文字は基本的に日本語で表記できるようにしてください。

## ステータス定義 (Status Definitions)
開発状況を追跡するため、各項目の先頭に以下のタグを付与してください。

*   `[Status: Todo]` : 未着手。AIが着手可能。
*   `[Status: WIP]` : (Work In Progress) 現在作業中。
*   `[Status: Review]` : 実装完了、人間のレビューまたはテスト待ち。
*   `[Status: Done]` : 完了・マージ済み。
*   `[Status: Blocked]` : 技術的課題や依存関係により停止中。
*   `[Status: Deferred]` : 次期フェーズへ延期
*   `[Test: Pending]` : テスト未作成。
*   `[Test: Pass]` : 検証済み。
*   `[Test: Fail]` : テスト失敗。修正が必要。
*   `[Test: Skip]` : 特定の理由でテストを除外中。
*   `[Known Issue]` : バグを認識した上でマージした項目。

## 1. 概要 (Overview)

Duel Masters AI Simulatorは、C++による高速なゲームエンジンと、Python/PyTorchによるAlphaZeroベースのAI学習環境を統合したプロジェクトです。

現在、**Phase 1: Game Engine Reliability** および **Phase 6: Engine Overhaul** の実装が完了し、C++コア（`dm_ai_module`）のビルドは安定しています。しかし、大規模なリファクタリング（EffectResolver削除、ディレクトリ移動）に伴い、**Pythonバインディングの不整合** が多数発生しており、テストスイートの半数以上が失敗している状態です。

直近の最優先課題は、**「Python Integration Repair（バインディング修復）」** を完了させ、既存のテストケースを全てパスさせることです。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` を削除し、`GameLogicSystem` へ完全移行しました。
*   [Status: Done] **GameLogicSystem Refactor**: `PipelineExecutor` を介したコマンド実行フローが確立されました。
*   [Status: Done] **Action Generalization**: 全アクションハンドラーの `compile_action` 化が完了しました。
*   [Status: Done] **Build Stability**: `cmake` によるビルドは警告のみで成功し、`dm_ai_module.so` が正常に生成されています。

### 2.1.1 実装済みメカニクス (Mechanics Support)
*   [Status: Done] **Revolution Change**: `tests/test_integrated_mechanics.py` にて検証済み（ただし現在はバインディングエラーで失敗中）。
*   [Status: Done] **Hyper Energy**: `CardKeywords.hyper_energy` 実装済み。
*   [Status: Done] **Just Diver**: `CardKeywords.just_diver` 実装済み。
*   [Status: Done] **Meta/Counter**: `tests/test_meta_counter.py` 実装済み。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Directory Restructuring**: `python/gui` を `dm_toolkit/gui` へ移動しました。
*   [Status: Done] **Encoding Audit**: ソースコードのShift-JIS対応完了。
*   [Test: Skip] **GUI Tests**: 現在のテスト環境（Headless）には `PyQt6` が含まれていないため、`tests/test_action_form_syntax.py` 等のGUIテストは実行不能（Skip/Fail）です。

### 2.3 AI & 学習基盤 (`src/ai` & `dm_toolkit/training`)
*   [Status: Done] **Directory Restructuring**: `python/training` を `dm_toolkit/training` へ移動しました。
*   [Status: Done] **Transformer Integration**: `NetworkV2` 実装完了。
*   [Status: Review] **Search Engine (MCTS)**: `src/ai/mcts` 実装完了。
*   [Status: Review] **Inference Engine**: `src/ai/inference` (PIMC, Deck Inference) 実装完了。
*   [Status: Review] **Lethal Solver**: `src/ai/solver` 実装完了 (Strict Implementation)。ただし `verify_lethal_puzzle.py` はバインディング欠落により失敗中。
*   [Status: Review] **POMDP**: `src/ai/pomdp` 実装完了。
*   [Status: WIP] [Test: Fail] **Deck Evolution**: `DeckEvolution` クラスのバインディング欠落により `tests/test_deck_evolution_cpp.py` が失敗しています。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Python Integration Repair (バインディングとテストの修復)
[Status: WIP] [Test: Fail]
テスト実行（`pytest`）の結果、以下のバインディング欠落が特定されました。これらを `src/bindings/bindings.cpp` に追加・修正する必要があります。

*   **Missing Attributes / Methods (C++ -> Python)**:
    *   `GameState` helper methods: `add_card_to_deck`, `add_card_to_hand`, `add_card_to_mana`, `add_test_card_to_battle` が欠落しており、多数のテスト（`test_atomic_actions.py`等）が失敗。
    *   `dm_ai_module.DeckEvolution`: `test_deck_evolution_cpp.py` で必須。
    *   `dm_ai_module.TriggerManager`: `test_phase6_reaction.py` で必須。
    *   `dm_ai_module.get_pending_effects_info`: `test_trigger_stack.py` で必須。
    *   `ScenarioConfig.my_deck`: `test_scenario.py` で必須。
    *   `dm_ai_module.register_batch_inference_numpy`: `verify_performance.py` で必須。
    *   `dm_ai_module.set_sequence_batch_callback`: `verify_transformer_support.py` で必須。

*   **Logic Disconnect**:
    *   `test_variable_system.py`: アクション（ドロー等）が実行されていません（Expected 3, got 0）。`PipelineExecutor` のコンテキスト変数の受け渡し（`$` prefixの処理など）に問題がある可能性があります。

### 3.2 [Priority: High] Phase 1: ゲームエンジンの信頼性 (Game Engine Reliability)
[Status: WIP]
エンジンのコアロジック自体は実装されていますが、テストを通じた検証が完了していません。

*   **Test Suite Status (2025/01 Consolidation Update)**:
    *   Total: ~84 tests (collected items)
    *   Passing: 49
    *   Failing: 26
    *   Errors: 2 (Import/Binding Errors)
    *   Skipped: 7 (PyQt6 or explicit skip)

### 3.3 [Priority: High] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Done]
C++側のリファクタリングは完了しました。現在はPython側からの制御（バインディング）のみが課題です。

### 3.5 [Priority: High] Test Directory Reorganization (テストディレクトリの再編)
[Status: Done]
`python/tests/` の内容を `tests/` に完全に移動・統合し、`python/tests/` ディレクトリを削除しました。
現在は `tests/` 直下のフラットな構造で運用されており、`pytest.ini` もこれに合わせて更新済みです。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Fix C++ Include Paths**: 相対パスインクルードの修正。
2.  [Status: WIP] [Test: Fail] **Debug Spell Pipeline**: `test_variable_system.py` や `verify_pipeline_spell.py` の失敗原因（ロジック不発）の究明。
3.  [Status: Done] **Encoding Audit**: 完了。
4.  [Status: Done] **Optimization - Shared Pointers**: 完了。
5.  [Status: WIP] **Verify Integration**: バインディング修復中。
6.  [Status: WIP] **Execution Logic Debugging**: パイプライン連携のデバッグ中。
7.  [Status: Done] **Memory Management**: 完了。
8.  [Status: Done] **Architecture Switch**: 完了。
9.  [Status: Done] **Transformer Verification**: 完了。
10. [Status: Todo] **Phase 7 Implementation**: 新JSONスキーマへの移行。
11. [Status: WIP] [Test: Fail] **Reaction Logic Integration**: `TriggerManager` バインディング待ち。
12. [Status: WIP] [Test: Fail] **Binding Restoration**: 最優先対応中。
13. [Status: Todo] **Verification Script Repair**: `verify_scenario_cpp.py` のパス修正。

## 5. テスト標準と運用要件 (Standard Testing Requirements)

### 5.1 ユニットテスト (Unit Tests)
*   **実行コマンド**: `PYTHONPATH=bin python3 -m pytest tests/`
    *   `bin` ディレクトリに `dm_ai_module.so` が存在することを確認してください。
    *   `tests/` ディレクトリ配下の全テストを実行します。
    *   `PyQt6` 依存テストはGUI環境がない場合スキップまたは失敗します。

### 5.2 シナリオ検証 (Scenario & Integration)
*   `tests/verify_lethal_puzzle.py`: リーサル計算検証。
*   `dm_toolkit/training/verify_performance.py`: AIパフォーマンス検証（要 `register_batch_inference_numpy`）。

### 5.3 運用ルール (Operational Rules)
1.  **コミット前検証**: `pytest` を実行し、既存のPass数（49）を下回らないことを確認する。
2.  **バインディング追従**: C++変更時は必ず `src/bindings/bindings.cpp` を更新する。
3.  **ドキュメント更新**: ステータスに変更があった場合は本ドキュメントを更新する。
