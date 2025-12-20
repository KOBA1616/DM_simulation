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

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** が完了し、**Phase 8: AI Architecture** の実装が完了しました。TransformerベースのAIモデル（NetworkV2）と、C++による高速なトークン化システムの統合フェーズに移行しています。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
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
*   [Status: Done] **Transformer Integration**: `NetworkV2` (DuelTransformer) クラス実装済み。Synergy Bias MaskおよびMeta-Game Embeddingを搭載。
*   [Status: Done] **Tokenization System**: C++側 (`TokenConverter`) での全ゾーン（Hand, Board, Grave, Mana）のトークン化、および語彙衝突回避（Vocabulary Offset）の実装完了。Pythonバインディングによる疎通確認済み。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: Review]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行しました。

*   **Step 1: イベント駆動基盤の実装**
    *   [Status: Done] [Test: Pass]
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   [Status: Done] [Test: Pass]
    *   `PipelineExecutor` (VM) を実装済み。スタックベース実行によるResume対応完了。
    *   `GAME_ACTION` 命令を追加し、高レベルなゲーム操作（プレイ、攻撃、ブロック）および勝敗判定をパイプライン経由で実行可能にしました。
*   **Step 3: GameCommand への統合**
    *   [Status: Done] [Test: Pass]
    *   `EffectResolver` の主要メソッドを `GameLogicSystem` へ移行。
*   **Step 4: Pure Command Generation (Current Focus)**
    *   [Status: Review] 各アクションハンドラー (`IActionHandler`) を `compile_action()` メソッドに対応させ、状態直接操作から命令生成へ移行しました。
    *   `DrawHandler`, `DiscardHandler`, `ModifierHandler`: 完了。
    *   `ManaHandler`, `DestroyHandler`, `SearchHandler`: 実装完了。
    *   [Status: Done] **Execution Logic**: `ADD_MANA`, `SEARCH_DECK`, `DESTROY` の実行時挙動修正が完了し、統合テスト (`tests/test_integration_pipeline.py`) がパスすることを確認しました。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 & データ移行
[Status: Pending]
全てのデータを新形式へ移行します。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   [Status: Done] [Test: Pass]
    *   JSONスキーマに `CommandDef` を導入済み。
*   **Step 2: CommandSystem の実装**
    *   [Status: Done] [Test: Pass]
    *   `dm::engine::systems::CommandSystem` を実装。

### 3.3 [Priority: Future] Phase 8: AI思考アーキテクチャの強化 (Advanced Thinking)
[Status: Review]
AIが「人間のような高度な思考（読み、コンボ、大局観）」を獲得するため、NetworkV2（Transformer）に対して以下の機能拡張および特徴量実装を行います。

*   **基本コンセプト (Core Concept)**:
    *   人間が「このカードは強い」といったヒューリスティックを与えるのではなく、**「構造（Structure）」と「素材（Raw Data）」を与え、AIが自己対戦を通じて意味と重みを学習（End-to-End Learning）できる設計**にします。
    *   **Implementation Status**: `DuelTransformer` クラス (`python.ai.models.transformer_model`) および C++トークナイザー (`dm_ai_module.TokenConverter`) を実装済み。
    *   **Tokenization Logic**: 全ゾーン（Hand, Board, Grave, Mana, History）を対象としたシーケンストークン化を実装。数値特徴量（Power, Cost, Turn）のバケット化およびOffsetによる語彙衝突回避を実装済み。
    *   **Model Logic**: Synergy Bias Matrix（カード間相性学習）およびMeta-Game Embedding（環境メタ学習）をTransformerのAttention機構に組み込み済み。

*   **実装要件 (Implementation Requirements)**

    #### A. 入力特徴量と次元圧縮 (Input Features & Compression)
    入力は固定長ベクトルではなく、複数のトークン系列（Sequence）として構成し、Transformerに入力します。

    1.  **[Feature 1] Action History (アクション履歴)**
        *   [Implemented] 過去のアクションコマンドをトークン列として入力。`MAX_HISTORY_LEN=128`。
    2.  **[Feature 2] Phase Tokens (フェイズトークン)**
        *   [Implemented] グローバル特徴量の一部として実装済み。
    3.  **[Feature 4, 6] Imperfect Info & Key Cards (不完全情報と重要カード)**
        *   [Implemented] `TokenConverter` により全カード情報をトークン化（ID + Status）。AIの視点（Perspective）に応じた情報マスキングは今後の課題。
    4.  **[Feature 7] Synergy Bias Mask (学習可能シナジー行列)**
        *   [Implemented] `DuelTransformer` 内に `synergy_matrix` [Vocab x Vocab] を実装し、Attention Maskとして適用。
    5.  **[Feature 8] Entity-Centric Board Token (詳細盤面トークン)**
        *   [Implemented] クリーチャーごとに ID, Tapped, Sick, Power, Cost, Civ をバケット化してシーケンス化。
    6.  **[Feature 9] Combo Completion (コンボ達成度)**
        *   [Pending] Cross-Attentionの実装は未着手。現在はSelf-Attentionによる暗黙的学習に依存。
    7.  **[Feature 12, 15] Meta-Game Context (メタゲーム)**
        *   [Implemented] `meta_embedding` トークンをシーケンス先頭に付与して実装済み。

    #### B. Neural Network Architecture (Model Config)
    Phase 4/8 で採用する Transformer (NetworkV2) の具体的な構成要件。
    *   **Architecture**: Encoder-Only Transformer (BERT-like)
    *   **Embedding Size ($d_{model}$)**: 256
    *   **Layers ($N_{layers}$)**: 4 (Prototype) -> 6 (Production)
    *   **Attention Heads ($h$)**: 8
    *   **Vocabulary Size**: 4096 (Implemented)
    *   **Context Length**: 2048 tokens (Implemented)

    #### C. Hyperparameters (Search & Training)
    AIの強さを決定づける探索および学習パラメータのベースライン要件。
    *   **MCTS Settings**:
        *   `num_simulations`: 800 (Training), 1600+ (Evaluation/Tournament)
        *   `c_puct`: 1.25 (Base exploration)
        *   `root_dirichlet_alpha`: 0.3 (For 30-40 legal moves)
        *   `root_exploration_fraction`: 0.25
    *   **Training Config**:
        *   `batch_size`: 512 - 1024
        *   `optimizer`: AdamW (`betas=(0.9, 0.999)`, `weight_decay=1e-4`)
        *   `lr_schedule`: Warmup (1000 steps) -> Cosine Decay

    #### D. Evolution Strategy (PBT Requirements)
    自己進化（Phase 3）における集団学習の要件。
    *   **Population Size**: 4 - 8 agents in parallel
    *   **Evaluation Metric**: Elo Rating (vs Past Versions & Fixed Baselines)
    *   **Gating**: 勝率 55% 以上で新世代として認定

## 4. 今後の課題 (Future Tasks)

1.  [Status: Done] **Optimization - Shared Pointers**: `GameState` のバインディングを `shared_ptr` 化し、不要なディープコピーを排除しました。
2.  [Status: Done] **Verify Integration**: ビルドおよびバインディングの修正が完了し、モジュールのインポートが可能になりました。
3.  [Status: Done] **Execution Logic Debugging**: `PipelineExecutor` を介したアクション実行ロジックの修正が完了し、統合テストがパスすることを確認しました。
4.  [Status: Done] **Memory Management**: `GameState` や `GameCommand` の所有権管理（`shared_ptr`）を一貫させ、メモリリークのリスクを大幅に低減しました。
5.  [Status: Done] **Architecture Switch**: `EffectResolver` の廃止により、`GameLogicSystem` と `PipelineExecutor` を中心としたコマンド実行アーキテクチャへの移行が完了しました。
6.  [Status: Review] **Transformer Implementation**: `DuelTransformer` と `TokenConverter` の実装が完了し、基本動作確認（Verify）をパスしました。今後は学習パイプライン (`train_simple.py`) への組み込みと、大規模学習による性能評価が必要です。
    *   **Issue 1: Perspective Logic**: 現在のトークナイザーは常にPlayer 1のカードを先にリストします。AIがPlayer 2としてプレイする場合の視点反転（Canonical View）の実装が必要です。
    *   **Issue 2: Token Spacing**: 一部のGlobal Feature（マナ数とシールド数など）のバケット範囲が隣接しており、モデルが混同する可能性があります（Positional Encodingで緩和されますが、明示的なスペーサートークン推奨）。
7.  [Status: Todo] **Phase 7 Implementation**: 新JSONスキーマへのデータ移行と、`CommandSystem` を利用した新フォーマットでのカード定義・実行の本格運用。

#### 新エンジン対応：Card Editor GUI構造の再定義

    新エンジン（イベント駆動・コマンド型）への移行に伴い、Card EditorのGUI（木構造）は、単なる「トリガー→アクション」のリストから、**「イベントリスナー」と「状態修飾子（Modifier）」を明確に区別する構造**へ変化させる必要があります。

    ##### 推奨される新しい木構造

    現在の `Card -> Effect -> Action` という3層構造を維持しつつ、**第2層（Effect層）の役割を拡張・分岐**させます。

    ```text
    [Root] Card Definition (基本情報: コスト、文明、種族など)
     │
     ├── [Node Type 1] Keywords (単純なキーワード能力)
     │    │  ※ 「ブロッカー」「W・ブレイカー」「SA」などのフラグ管理
     │    └─ (Checkboxes / List)
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
