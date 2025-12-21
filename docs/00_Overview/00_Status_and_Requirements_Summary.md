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

ディレクトリ構造の再編を行い、Python側のGUIおよび学習コードを `dm_toolkit` パッケージへ移動しました。現在、この変更に伴う **バインディングおよびテストの不整合** が多数発生しており、これらの修復が最優先事項となっています。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: WIP] [Test: Fail] **Lethal Solver Strict Implementation**: `GameState` API変更（`clear_zone`削除等）によりテスト `tests/python/test_lethal_solver.py` が失敗しており、改修が必要です。
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` に加え、`Stat` (統計更新), `GameResult` (勝敗判定), `Attach` (進化/クロス) を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令 (`WIN_GAME`, `LOSE_GAME`, `TRIGGER_CHECK`, `STAT`更新) をサポートするように拡張されました。
*   [Status: Done] **Stack-Based VM**: `PipelineExecutor` を再帰ベースからスタックベースのVM（Virtual Machine）にリファクタリングし、`Instruction` 実行の一時停止と再開（Resume）を完全にサポートしました。
*   [Status: Done] **Complete Effect Resolution**: `EffectSystem::compile_effect` を実装し、複雑な効果（S・トリガー等）を `Instruction` リストにコンパイルして実行するフローを確立しました。
*   [Status: Done] **Action Generalization**: すべての `IActionHandler` (合計20クラス以上) に `compile_action` メソッドを追加し、インターフェースを統一しました。
*   [Status: Done] **Optimization - Shared Pointers**: `GameState` のPythonバインディングにおけるディープコピー問題を解消するため、`ParallelRunner`、`MCTS`、`GameResultInfo`、および `bindings.cpp` を `std::shared_ptr<GameState>` ベースに移行しました。
*   [Status: Done] **Handler Migration**: `ManaHandler`, `DestroyHandler`, `SearchHandler` などの実装を `compile_action` ベースに移行しました。
*   [Status: Done] **Build Fixes**: C++コアのビルドは成功し、`dm_ai_module.so` は生成されています。ただし、Pythonバインディングとの整合性に問題が残っています。

### 2.1.1 実装済みメカニクス (Mechanics Support)
*   [Status: Done] **Revolution Change**: `CardDefinition` への `revolution_change` フラグおよび `revolution_change_condition` の追加、及び統合テストでの動作確認完了。
*   [Status: Done] **Hyper Energy**: `CardKeywords.hyper_energy` および UI サポート実装済み。
*   [Status: Done] **Just Diver**: `CardKeywords.just_diver` 実装済み。
*   [Status: Done] **Meta/Counter (Hand Triggers)**: `tests/python/test_meta_counter.py` 等で基盤実装済み。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Directory Restructuring**: `python/gui` を `dm_toolkit/gui` へ移動しました。
*   [Status: Done] **Encoding Audit**: `dm_toolkit/gui/` 内の全Pythonソースコードに `coding: cp932` (Shift-JIS) 宣言を追加し、Windows環境での動作安定性を確保しました。
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。`CardEditForm` は Revolution Change や新キーワードに対応済み。
*   [Status: Done] **Frontend Integration**: GUI (`app.py`) が `waiting_for_user_input` フラグを監視し、対象選択やオプション選択ダイアログを表示してゲームループを再開（Resume）する機能を実装しました。
*   [Status: Done] **Data Structure Update**: 新エンジンの仕様に合わせて、GUI上のデータ構造を以下の3層に明確化しました。
    *   **Keywords (Type 1)**: 単純なキーワード能力（ブロッカー等）を `KeywordEditForm` で管理。
    *   **Abilities (Type 2)**: 誘発型能力（Triggered）と常在型能力（Static）をグループ化して表示。
    *   **Reaction Abilities (Type 3)**: ニンジャ・ストライクなどのリアクション能力専用のノードを追加。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Directory Restructuring**: `python/training` を `dm_toolkit/training` へ移動しました。
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Done] **Transformer Integration**: `NetworkV2` (DuelTransformer) クラスが実装され、`verify_performance.py` への統合が完了しました。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Python Integration Repair (バインディングとテストの修復)
[Status: WIP] [Test: Fail]
C++エンジンの更新に対し、Pythonバインディングおよびテストコードの追従が遅れており、多くの検証スクリプトが動作不能となっています。

*   **Missing / Incompatible Bindings**:
    *   `GameEvent`: インポートエラー (`test_phase6_reaction.py`)
    *   `CardDefinition`: コンストラクタ引数の不一致 (`verify_trigger_system.py`)
    *   `CommandDef` / `CommandType`: 定義不足または名称不一致 (`verify_query_command.py`)
    *   `CardRegistry`: シングルトンアクセス方法の不一致 (`verify_continuous_recalc.py`)
    *   `Instruction`: コンストラクタ引数の不一致 (`test_complex_instructions.py`)
*   **Module Structure**: `dm_toolkit` への移動に伴い、テストコード内のインポートパスおよび `PYTHONPATH` の設定修正が必要です。

### 3.2 [Priority: High] Phase 1: ゲームエンジンの信頼性 (Game Engine Reliability)
[Status: WIP]
テストカバレッジの向上とバグ修正を最優先します。

*   **Test Suite Restoration**:
    *   [Status: Done] `test_pomdp_parametric.py`, `test_scenario.py` のバインディング不足を解消しパスさせました。
    *   [Status: WIP] [Test: Fail] `test_variable_system.py`: `ActionDef` バインディングの属性不足 (`target_choice`) により失敗中。
    *   [Status: WIP] [Test: Fail] `verify_pipeline_spell.py`: 呪文解決ロジック（墓地送り、効果発動）が正しく動作しておらず、失敗しています。
    *   **Next Action**: バインディング修正後、全てのテスト (`tests/python/` および `tests/`) を実行し、回帰テストを行う。

### 3.3 [Priority: High] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Done]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行しました。
C++側の実装は完了していますが、前述のバインディング問題によりPython側からの制御に支障が出ています。

### 3.4 [Priority: Future] Phase 8: AI思考アーキテクチャの強化 (Advanced Thinking)
[Status: WIP]
AIが「人間のような高度な思考（読み、コンボ、大局観）」を獲得するため、NetworkV2（Transformer）に対して以下の機能拡張および特徴量実装を行います。

*   **基本コンセプト (Core Concept)**:
    *   人間が「このカードは強い」といったヒューリスティックを与えるのではなく、**「構造（Structure）」と「素材（Raw Data）」を与え、AIが自己対戦を通じて意味と重みを学習（End-to-End Learning）できる設計**にします。
    *   **Implementation Status**: `DuelTransformer` クラス (`dm_toolkit.ai.agent.transformer_model`) および `NetworkV2` (`dm_toolkit.training.network_v2`) を実装済み。

### 3.5 [Priority: High] Test Directory Reorganization (テストディレクトリの再編)
[Status: Done]
新エンジン対応の単一のテストコードベースを確立するため、分散していたテストディレクトリを統合しました。

*   **Changes**:
    *   `python/tests/` および `tests/python/` の内容を全て `tests/` ルートへ移動しました。
    *   `pytest.ini` の `testpaths` を `tests` に変更しました。
    *   今後は全てのテストコードを `tests/` 直下に配置し、新エンジン（`dm_ai_module`）に対する検証を一元管理します。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Fix C++ Include Paths**: `src/ai/encoders/token_converter.hpp` および `src/utils/csv_loader.hpp` に存在する相対パスインクルード（`../../` 等）をプロジェクト標準の `src/` 起点の絶対パスに修正する。
2.  [Status: WIP] [Test: Fail] **Debug Spell Pipeline**: 統合テスト `tests/verify_pipeline_spell.py` が失敗しています。`ActionGenerator` または `EffectResolver` (GameLogicSystem) の呪文処理ロジックの再検証が必要です。
3.  [Status: Done] **Encoding Audit**: `dm_toolkit/gui/` 内のPythonソースコードに `coding: cp932` (Shift-JIS) 宣言を追加し、日本語環境での表示・実行時の不具合を防止しました。
4.  [Status: Done] **Optimization - Shared Pointers**: `GameState` のバインディングを `shared_ptr` 化し、不要なディープコピーを排除しました。
5.  [Status: Done] **Verify Integration**: ビルドおよびバインディングの修正が完了し、モジュールのインポートが可能になりました。
6.  [Status: Done] **Execution Logic Debugging**: `PipelineExecutor` を介したアクション実行ロジックの修正が完了し、統合テストがパスすることを確認しました。
7.  [Status: Done] **Memory Management**: `GameState` や `GameCommand` の所有権管理（`shared_ptr`）を一貫させ、メモリリークのリスクを大幅に低減しました。
8.  [Status: Done] **Architecture Switch**: `EffectResolver` の廃止により、`GameLogicSystem` と `PipelineExecutor` を中心としたコマンド実行アーキテクチャへの移行が完了しました。
9.  [Status: Done] **Transformer Verification**: 実装された `DuelTransformer` の学習パフォーマンス検証と、完全トークン化された入力特徴量への移行が完了しました。
10. [Status: Todo] **Phase 7 Implementation**: 新JSONスキーマへのデータ移行と、`CommandSystem` を利用した新フォーマットでのカード定義・実行の本格運用。
11. [Status: WIP] [Test: Fail] **Reaction Logic Integration**: リアクション能力（Node Type 3）のテスト `tests/test_phase6_reaction.py` がバインディング（`GameEvent`等）の不足により失敗しています。エンジン側での完全な実行ロジックの実装と検証が必要です。
12. [Status: WIP] [Test: Fail] **Binding Restoration**: `CardDefinition` コンストラクタ、`Instruction` コンストラクタ、`CardRegistry` シングルトンなどのPythonバインディング不整合を解消し、テストスイートを復旧させる必要があります。

#### 新エンジン対応：Card Editor GUI構造の再定義

    新エンジン（イベント駆動・コマンド型）への移行に伴い、Card EditorのGUI（木構造）は、単なる「トリガー→アクション」のリストから、**「イベントリスナー」と「状態修飾子（Modifier）」を明確に区別する構造**へ変化させる必要があります。

    [Status: Done] 以下の構造への移行を完了しました。

    ```text
    [Root] Card Definition (基本情報: コスト、文明、種族など)
     │
     ├── [Node Type 1] Keywords (単純なキーワード能力)
     │    │  ※ 「ブロッカー」「W・ブレイカー」「SA」などのフラグ管理
     │    └─ (Checkboxes / List) -> KeywordEditForm
     │
     ├── [Node Type 2] Abilities (複雑な能力リスト)
     │    │
     │    ├── Case A: Triggered Ability (誘発型能力 / イベントリスナー)
     │    │    │  ※ 特定のイベントに反応してスタックに乗る能力
     │    │    │
     │    │    ├── Trigger Definition (いつ？)
     │    │    │    └─ Event Type: ON_PLAY, ON_ATTACK, ON_DESTROY, ON_BLOCK
     │    │    │
     │    │    ├── Condition (条件は？ - 介入型if節)
     │    │    │    └─ Filter: "If you have a Fire Bird", "If opponent has no shields"
     │    │    │
     │    │    └── Action Sequence (何をする？ - コマンド発行)
     │    │         ├── Action 1: SELECT_TARGET (Targeting)
     │    │         └── Action 2: DESTROY_CARD (Execution)
     │    │
     │    └── Case B: Static / Passive Ability (常在型能力 / 状態修飾)
     │         │  ※ 戦場にある限り常に適用される効果（Modifer）
     │         │
     │         ├── Layer Definition (何を変える？)
     │         │    └─ Type: COST_MODIFIER, POWER_MODIFIER, GRANT_KEYWORD
     │         │
     │         ├── Condition (適用条件は？)
     │         │    └─ Filter: "While tapped", "If mana > 5"
     │         │
     │         └── Value / Target Scope (誰に・どれくらい？)
     │              └─ Target: ALL_FRIENDLY_CREATURES, SELF
     │              └─ Value: +3000, -1 Cost
     │
     └── [Node Type 3] Reaction Abilities (リアクション / 忍者ストライク等)
          │  ※ 手札などから特定のタイミングで宣言できる能力
          │
          ├── Trigger Window (どのタイミングで？)
          │    └─ Type: NINJA_STRIKE, STRIKE_BACK
          │
          └── Action Sequence
               └─ (Summon, Cast, etc.)
    ```

## 5. テスト標準と運用要件 (Standard Testing Requirements)

本プロジェクトの品質を担保するため、以下のテスト群を標準テストスイートとして定義し、開発プロセスにおいて必ず実行・維持するものとします。

### 5.1 ユニットテスト (Unit Tests)
C++エンジン、バインディング、および基本ルールロジックの整合性を検証します。
*   **実行コマンド**: `pytest tests/`
*   **必須パス基準**: `tests/test_*.py` に分類される全てのテストケース。

### 5.2 シナリオ検証 (Scenario & Integration)
複雑な盤面状態や特定のメカニクス（Lethal、ループ、特殊勝利）を検証する統合テストです。
*   `tests/verify_lethal_puzzle.py`: 詰将棋的なリーサル計算の正確性検証。
*   `tests/verify_pipeline_spell.py`: 呪文処理パイプラインの動作検証。
*   `tests/verify_trigger_system.py`: トリガー連鎖および解決順序の検証。
*   `tests/verify_performance.py`: AI思考エンジンの動作確認および基本パフォーマンス計測。

### 5.3 運用ルール (Operational Rules)
1.  **コミット前検証**: 変更を加えた際は、関連するユニットテストおよび `pytest` 全体を実行し、回帰（Regression）がないことを確認する。
2.  **バインディング追従**: C++側のAPI変更を行った場合、即座にPythonバインディングを更新し、`test_binding.py` 等で不整合がないか確認する。
3.  **ドキュメント更新**: テスト環境や実行方法に変更があった場合、本ドキュメントおよび `AGENTS.md` を更新する。
