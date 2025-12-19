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

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** の最終段階にあり、アクションハンドラーの `compile_action` 化とビルド修正を集中的に行っています。

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
*   [Status: Review] **Handler Migration**: `ManaHandler`, `DestroyHandler`, `SearchHandler` の実装を `compile_action` ベースに移行しました。これにより、これらのアクションは直接的な状態変更ではなく、`Instruction` 列の生成として機能します。
*   [Status: Review] **Build Fixes**: `GameState` の非コピー可能性（`unique_ptr`の使用による）に起因するビルドエラーを修正しました。`EffectSystem` のシングルトン参照渡しや `bindings.cpp` のラムダ式活用を行いました。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Done] **Frontend Integration**: GUI (`app.py`) が `waiting_for_user_input` フラグを監視し、対象選択やオプション選択ダイアログを表示してゲームループを再開（Resume）する機能を実装しました。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Blocked] **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

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
    *   `ManaHandler`, `DestroyHandler`, `SearchHandler`: 実装完了、ビルド検証待ち。
    *   **Complex Handlers Migration**: `RevealHandler`, `SelectOptionHandler` などの複雑なハンドラーもインターフェース上は移行済みですが、実装詳細の検証が必要です。

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
[Status: Deferred]
AIが「人間のような高度な思考（読み、コンボ、大局観）」を獲得するため、NetworkV2（Transformer）に対して以下の機能拡張および特徴量実装を行います。

*   **基本コンセプト (Core Concept)**:
    *   人間が「このカードは強い」といったヒューリスティックを与えるのではなく、**「構造（Structure）」と「素材（Raw Data）」を与え、AIが自己対戦を通じて意味と重みを学習（End-to-End Learning）できる設計**にします。
    *   LLMによる言語化や感想戦機能を導入し、ブラックボックス化を防ぎます。

*   **実装要件 (Implementation Requirements)**

    #### A. 入力特徴量と次元圧縮 (Input Features & Compression)
    入力は固定長ベクトルではなく、複数のトークン系列（Sequence）として構成し、Transformerに入力します。

    1.  **[Feature 1] Action History (アクション履歴)**
        *   過去 $N$ ターン分の「プレイ」「チャージ」「攻撃」を時系列トークンとして入力。
        *   アクション数上限: **過去10ターン分**、かつ各ターンの主要アクション（Play, Charge, Attack）に絞り込み、最大 **30トークン程度** に制限してノイズを抑制する。
        *   目的: 「手札温存（ブラフ）」やデッキタイプ推定などの文脈理解。
    2.  **[Feature 2] Phase Tokens (フェイズトークン)**
        *   `[MANA]`, `[MAIN]`, `[ATTACK]` 等の特殊トークンを系列先頭に付与。
        *   目的: フェイズごとにAttentionの注目先（マナカーブ、盤面処理など）を切り替える。
    3.  **[Feature 4, 6] Imperfect Info & Key Cards (不完全情報と重要カード)**
        *   **Key Card Count**: 公開領域にある全カードIDの枚数ヒストグラム（2000次元）をLinear圧縮して1トークン化。
        *   **Hidden Inference**: 相手の手札/シールドにある確率分布（2000次元）をLinear圧縮して1トークン化。
        *   目的: 「ボルメテウスが墓地に落ちた」等の重要イベントを、人間が指定せずともAIが勝率との相関から学習する。
    4.  **[Feature 7] Synergy Bias Mask (学習可能シナジー行列)**
        *   カードID間の相性（$N \times N$）を表す学習可能な行列 (`SynergyMatrix`) を導入し、Attentionスコアに加算。
        *   目的: 種族や文明を超えた「実戦的なコンボ相性」を自動獲得させる。
    5.  **[Feature 8] Entity-Centric Board Token (詳細盤面トークン)**
        *   バトルゾーンのクリーチャーを「ID + パワー(生数値) + フラグ(ブロッカー等)」の結合ベクトルとしてトークン化。
        *   目的: 「パワー6000以上」といった閾値を人間が決めず、AIに生の数値から脅威度を判断させる。
    6.  **[Feature 9] Combo Completion (コンボ達成度)**
        *   Cross-Attention (手札トークン列 vs 盤面トークン列) を導入。
        *   目的: Multi-Head Attentionに「進化元と進化先」などのペア関係を専門的に監視させ、コンボ成立を検知させる。
    7.  **[Feature 12, 15] Meta-Game Context (メタゲーム)**
        *   自分と相手のデッキタイプ（アーキタイプ）や、環境の流行を表すベクトルをLinear圧縮して入力。
        *   目的: 対面に応じたプレイスタイルの切り替え（アグロ対面なら防御優先など）。

    #### B. Neural Network Architecture (Model Config)
    Phase 4/8 で採用する Transformer (NetworkV2) の具体的な構成要件。
    *   **Architecture**: Encoder-Only Transformer (BERT-like)
    *   **Embedding Size ($d_{model}$)**: 256
    *   **Layers ($N_{layers}$)**: 6
    *   **Attention Heads ($h$)**: 8
    *   **Feed-Forward Network ($d_{ff}$)**: 1024
    *   **Context Length**: 512 tokens (Max)
    *   **Activation**: GELU

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
2.  [Status: Todo] **Verify Integration**: ビルド修正後、`tests/test_effect_compiler.py` や `tests/test_engine_basics.py` を実行し、新エンジンの動作を統合的に検証する。
3.  [Status: Todo] **Memory Management**: `GameState` や `GameCommand` の所有権管理（`unique_ptr` vs `shared_ptr`）を一貫させ、メモリリークや二重解放のリスクを排除する。
4.  [Status: Todo] **Architecture Switch**: `GameLogicSystem` が直接 `compile()` を呼び出し、生成された命令を一括実行するアーキテクチャへの完全切り替え。

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
    ##### 具体的な変更点と理由
    1.  **「Trigger」から「Ability Type」への概念拡張**
        *   これまでは全ての効果を「何かが起きたら実行する」として扱っていましたが、新エンジンでは**「即時解決されるイベント（誘発）」**と**「継続的に適用されるルール（常在）」**を区別する必要があります。
        *   **GUIの変更:** Effectを追加する際、最初に **「Triggered (誘発型)」** か **「Static (常在型)」** かを選択させます。
    2.  **条件判定（Condition）の独立ノード化**
        *   新エンジンでは、イベントが発生しても「条件を満たしていなければ PendingEffect (待機効果) を生成しない」あるいは「解決時に不発になる」という判定が重要です。
        *   **GUIの変更:** TriggerとActionの間に **「Condition (条件)」** ノードまたはプロパティ欄を設けます。
    3.  **アクション間の「変数のリンク（Context Linking）」**
        *   コマンド式になったことで、Action 1（選択）の結果を Action 2（破壊）が受け取るフローが厳格になります。
        *   **GUIの変更:** Action定義画面に **「Input Source」** という項目を追加し、前のActionの出力やイベント発生源を指定するUIを改善します。
