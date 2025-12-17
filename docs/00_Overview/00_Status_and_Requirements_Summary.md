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

現在、**Phase 6: Engine Overhaul (EffectResolverからGameCommandへの完全移行)** の最終段階にあり、**Phase 7: Data Migration (全カードデータの新JSONフォーマット移行)** への移行準備を進めています。
`Pure Command Generation` の基盤が整備され、`DrawHandler` などの主要ハンドラーが命令パイプライン形式に移行を開始しました。

## 2. 現行システムステータス (Current Status)

### 2.1 コアエンジン (C++ / `src/engine`)
*   [Status: Done] **EffectResolver Removal**: `EffectResolver` クラスおよびファイルを物理的に削除しました。すべての呼び出し元 (`GameInstance`, `ActionGenerator`, `ScenarioExecutor`, `Bindings`) は `GameLogicSystem` へ移行されました。
*   [Status: Done] **GameLogicSystem Refactor**: アクションディスパッチロジックを `GameLogicSystem` に集約し、`PipelineExecutor` を介した処理フローを確立しました。
*   [Status: Done] **GameCommand**: 新エンジンの核となるコマンドシステム。`Transition`, `Mutate`, `Flow` に加え、`Stat` (統計更新), `GameResult` (勝敗判定) を実装済み。
*   [Status: Done] **Instruction Pipeline**: `PipelineExecutor` が `GAME_ACTION` 命令 (`WIN_GAME`, `LOSE_GAME`, `TRIGGER_CHECK`, `STAT`更新) をサポートするように拡張されました。
*   [Status: WIP] **Pure Command Generation**: `EffectSystem` に `compile_action` メソッドを追加し、アクション定義から `Instruction` リストを生成する仕組みを実装しました。`DrawHandler` の移行が完了（デッキ切れ判定→移動→統計更新→トリガーチェックの命令生成）。
*   [Known Issue] **Binding SegFault**: `Instruction` 構造体の再帰的定義と `nlohmann::json` 引数の Python バインディングにおいて、複雑なオブジェクト受け渡し時に Segmentation Fault が発生する問題が確認されています (`tests/test_effect_compiler.py`)。

### 2.2 カードエディタ & ツール (`dm_toolkit/gui`)
*   [Status: Done] **Status**: 稼働中 (Ver 2.3)。
*   [Status: Deferred] **Freeze**: 新JSONスキーマが確定次第、新フォーマット専用エディタとして改修を行う。

### 2.3 AI & 学習基盤 (`dm_toolkit/training`)
*   [Status: Done] **Status**: パイプライン構築済み。
*   [Status: Blocked] **Pending**: エンジン刷新に伴うデータ構造の変更が確定するまで凍結。

---

## 3. ロードマップ (Roadmap)

### 3.1 [Priority: Critical] Phase 6: エンジン刷新 (Engine Overhaul)
[Status: WIP]
`EffectResolver` を解体し、イベント駆動型システムと命令パイプラインへ完全移行します。

*   **Step 1: イベント駆動基盤の実装**
    *   [Status: Done] [Test: Pass]
    *   `TriggerManager`: シングルトン/コンポーネントによるイベント監視・発行システムの実装。（実装完了）
*   **Step 2: 命令パイプライン (Instruction Pipeline) の実装**
    *   [Status: Done] [Test: Pass]
    *   `PipelineExecutor` (VM) を実装済み。
    *   `GAME_ACTION` 命令を追加し、高レベルなゲーム操作（プレイ、攻撃、ブロック）および勝敗判定をパイプライン経由で実行可能にしました。
*   **Step 3: GameCommand への統合**
    *   [Status: Done] [Test: Pass]
    *   `EffectResolver` の主要メソッドを `GameLogicSystem` へ移行。
*   **Step 4: Pure Command Generation (Current Focus)**
    *   [Status: WIP] 各アクションハンドラー (`IActionHandler`) を `compile()` メソッドに対応させ、状態直接操作から命令生成へ移行する。
    *   `DrawHandler`: 完了。
    *   `ManaHandler`: 未着手。
    *   `DestroyHandler`: 未着手。
    *   `SearchHandler`: 未着手。

### 3.2 [Priority: High] Phase 7: ハイブリッド・エンジン基盤 & データ移行
[Status: Pending]
全てのデータを新形式へ移行します。

*   **Step 1: データ構造の刷新 (Hybrid Schema)**
    *   [Status: Done] [Test: Pass]
    *   JSONスキーマに `CommandDef` を導入済み。
*   **Step 2: CommandSystem の実装**
    *   [Status: Done] [Test: Pass]
    *   `dm::engine::systems::CommandSystem` を実装。

### 3.3 [Priority: Future] Phase 8: Transformer拡張 (Hybrid Embedding)
[Status: Deferred]
Transformer方式を高速化し、かつZero-shot（未知のカードへの対応）を可能にするため、「ハイブリッド埋め込み (Hybrid Embedding)」を導入します。また、文脈として墓地のカードも対象に含めます。

*   **コンセプト (Concept)**
    *   **Hybrid Embedding**: `Embedding = Embed(CardID) + Linear(ManualFeatures)`
    *   **Zero-shot対応**: 未知のカード（ID埋め込み未学習）でも、スペック情報（コスト、文明、パワー等）から挙動を推論可能にする。
    *   **スコープ拡張**: Transformerの入力文脈に「墓地」のカードも含め、墓地利用や探索に対応させる。

*   **実装要件 (Requirements)**
    *   [Status: Todo] **A. C++側 (TensorConverter)**
        *   `convert_to_sequence` を修正し、`Output: (TokenSequence, FeatureSequence)` を返すように変更する。
    *   [Status: Todo] **B. Python側 (NetworkV2)**
        *   モデル入力層を修正: `x_id` (Card IDs) と `x_feat` (Manual Features) を受け取る。
    *   [Status: Todo] **C. 特徴量ベクトルの定義 (Feature Vector Definition)**
        *   **Card Function Embeddings (機能抽象化)**: 除去、ドロー、ブロッカー等の役割をベクトル化し、未知のカードに対応（Zero-shot）。
        *   **Projected Effective Mana (次ターン有効マナ)**: 次ターンのアンタップマナ数と発生可能文明を予測し、色事故を防ぐ。
        *   **Synergy/Combo Readiness (コンボ成立度)**: 進化元やコンボパーツの揃い具合を数値化し、戦略的キープを促進。
        *   **Turns to Lethal (リーサル距離)**: LethalSolverによる「勝利/敗北までのターン数」を入力し、終盤の判断力を強化。
        *   その他: 基本スペック（コスト、パワー、文明）、キーワード能力、リソース操作等。
    *   [Status: Todo] **D. Advanced Lethal Search (Mini-Max)**
        *   現行のGreedy Heuristicに加え、手札（SA、進化速攻）やトリガーリスクを考慮した「超短期探索型 (Mini-Max Search) LethalSolver」を実装する。
        *   エンジンをコピーしてシミュレーションを行うことで、攻撃時効果やブロッカー除去も正確に判定可能にする。

## 4. 今後の課題 (Future Tasks)

1.  [Status: Todo] **Binding Fix**: `Instruction` 構造体の Python バインディングにおけるメモリ管理・コピーの問題（SegFault）を解決する。
2.  [Status: Todo] **Handler Migration**: `ManaHandler`, `DestroyHandler` 等の主要ハンドラーを `compile()` パターンへ移行する。
3.  [Status: Todo] **EffectResolver Cleanup**: 残存する `EffectResolver` のロジックがあれば完全に `PipelineExecutor` へ統合する。

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
        *   **GUIの変更:** Action定義画面に **「Input Source」** という項目を追加し、前のActionの出力やイベント発生源を指定できるようにします。
