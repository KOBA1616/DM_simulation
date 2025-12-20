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

現在、**Phase 1: Game Engine Reliability (Lethal Solverの厳密化)** および **Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** の最終段階にあり、アクションハンドラーの `compile_action` 化とビルド修正を集中的に行っています。また、**Phase 8: AI Architecture** の実装が並行して開始されています。

直近では、Pythonバインディングの拡充（Phase 6）と、テストスイートの適合性向上（Phase 1）を実施しました。特に変数リンクシステム（Variable Linking）の修正に取り組んでいます。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **Binding Update**: `ParametricBelief`, `GameInstance` (再公開), `FilterDef.types`, `EffectType` (Enum) のPythonバインディングを追加・修正しました。これにより多数のテストケース (`test_pomdp_parametric.py`, `test_scenario.py`) がパスするようになりました。
*   [Status: WIP] **Variable Linking Fix**: `DrawHandler` と `PipelineExecutor` における変数参照（`$var_name`）と動的ループ（`REPEAT`）の処理を修正中です。`test_variable_system.py` でJSON型エラーが発生しており、解決に取り組んでいます。
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` に加え、`Stat` (統計更新), `GameResult` (勝敗判定), `Attach` (進化/クロス) を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令 (`WIN_GAME`, `LOSE_GAME`, `TRIGGER_CHECK`, `STAT`更新) をサポートするように拡張されました。
*   [Status: Done] **Stack-Based VM**: `PipelineExecutor` を再帰ベースからスタックベースのVM（Virtual Machine）にリファクタリングし、`Instruction` 実行の一時停止と再開（Resume）を完全にサポートしました。
*   [Status: Done] **Complete Effect Resolution**: `EffectSystem::compile_effect` を実装し、複雑な効果（S・トリガー等）を `Instruction` リストにコンパイルして実行するフローを確立しました。
*   [Status: Done] **Action Generalization**: すべての `IActionHandler` (合計20クラス以上) に `compile_action` メソッドを追加し、インターフェースを統一しました。
*   [Status: Done] **Optimization - Shared Pointers**: `GameState` のPythonバインディングにおけるディープコピー問題を解消するため、`ParallelRunner`、`MCTS`、`GameResultInfo`、および `bindings.cpp` を `std::shared_ptr<GameState>` ベースに移行しました。
*   [Status: Done] **Handler Migration**: `ManaHandler`, `DestroyHandler`, `SearchHandler` などの実装を `compile_action` ベースに移行しました。
*   [Status: Done] **Build Fixes**: `GameState` の非コピー可能性に起因するビルドエラーを修正し、`EffectSystem` のシングルトン参照渡しや `bindings.cpp` のラムダ式活用を行いました。Pythonバインディングの `InstructionOp` および `TriggerType` 不足も解消しました。

### 2.1.1 実装済みメカニクス (Mechanics Support)
*   [Status: Done] **Revolution Change**: `CardDefinition` への `revolution_change` フラグおよび `revolution_change_condition` の追加、及び統合テストでの動作確認完了。
*   [Status: Done] **Hyper Energy**: `CardKeywords.hyper_energy` および UI サポート実装済み。
*   [Status: Done] **Just Diver**: `CardKeywords.just_diver` 実装済み。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。`CardEditForm` は Revolution Change や新キーワードに対応済み。
*   [Status: Done] **Frontend Integration**: GUI (`app.py`) が `waiting_for_user_input` フラグを監視し、対象選択やオプション選択ダイアログを表示してゲームループを再開（Resume）する機能を実装しました。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Done] **Transformer Integration**: `NetworkV2` (DuelTransformer) クラスが実装され、`verify_performance.py` への統合が完了しました。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: High] Phase 1: ゲームエンジンの信頼性 (Game Engine Reliability)
[Status: WIP]
テストカバレッジの向上とバグ修正を最優先します。

*   **Test Suite Restoration**:
    *   [Status: Done] `test_pomdp_parametric.py`, `test_scenario.py` のバインディング不足を解消しパスさせました。
    *   [Status: WIP] `test_variable_system.py`: `DRAW_CARD` の変数リンク機能におけるJSON型エラーを修正中。
    *   **Next Action**: 修正後、全てのテスト (`tests/python/`) を実行し、回帰テストを行う。

### 3.2 [Priority: High] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Done]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行しました。
バインディング層のメンテナンス（今回実施分）により、Python側からの利用可能性が回復しました。

### 3.3 [Priority: Future] Phase 8: AI思考アーキテクチャの強化 (Advanced Thinking)
[Status: WIP]
AIが「人間のような高度な思考（読み、コンボ、大局観）」を獲得するため、NetworkV2（Transformer）に対して以下の機能拡張および特徴量実装を行います。

*   **基本コンセプト (Core Concept)**:
    *   人間が「このカードは強い」といったヒューリスティックを与えるのではなく、**「構造（Structure）」と「素材（Raw Data）」を与え、AIが自己対戦を通じて意味と重みを学習（End-to-End Learning）できる設計**にします。
    *   **Implementation Status**: `DuelTransformer` クラス (`dm_toolkit.ai.agent.transformer_model`) および `NetworkV2` (`dm_toolkit.training.network_v2`) を実装済み。

## 4. 今後の課題 (Future Tasks)

1.  [Status: WIP] **Fix Variable Linking**: `DrawHandler` と `PipelineExecutor` の連携を修正し、動的なドロー枚数指定を正常化する（`test_variable_system.py` のパス）。
2.  [Status: Todo] **Fix C++ Include Paths**: `src/ai/encoders/token_converter.hpp` および `src/utils/csv_loader.hpp` に存在する相対パスインクルード（`../../` 等）をプロジェクト標準の `src/` 起点の絶対パスに修正する。
3.  [Status: Todo] **Debug Spell Pipeline**: 統合テスト `tests/verify_pipeline_spell.py` の失敗原因（呪文カードが墓地へ移動せず、効果が発動しない問題）を調査し、`ActionGenerator` または `EffectResolver` (GameLogicSystem) の呪文処理ロジックを修正する。
4.  [Status: Todo] **Encoding Audit**: `dm_toolkit/gui/` 内のPythonソースコードに `coding: cp932` (Shift-JIS) 宣言を追加し、日本語環境での表示・実行時の不具合を防止する。
