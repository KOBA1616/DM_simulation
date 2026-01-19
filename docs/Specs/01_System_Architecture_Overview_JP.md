```markdown
# デュエル・マスターズ AI システムアーキテクチャ詳細概要

本ドキュメントは、プロジェクト「Duel Masters AI」の技術的な全体像、コアエンジンの仕組み、AIパイプライン、およびデータ構造について、開発者向けに詳細かつ網羅的に解説するものです。

## 1. システム全体像 (System Overview)

本プロジェクトは、**C++20**による高性能なゲームエンジンと、**Python**による柔軟なAI学習・GUI環境を組み合わせたハイブリッド構成を採用しています。

*   **C++ (Core Engine):** ゲームルール、状態管理、探索アルゴリズム（MCTS）、並列シミュレーションを担当。数万回の対戦を高速に実行可能です。
*   **Python (Wrapper & Tools):** C++コアへのバインディング（`pybind11`）を介して、ニューラルネットワークの学習（PyTorch）、データセット管理、GUIツール（カードエディタ、対戦可視化）を提供します。

### 技術スタック
*   **言語:** C++20 (Engine), Python 3.10+ (Scripting/ML)
*   **ビルド:** CMake
*   **バインディング:** pybind11
*   **AI:** PyTorch, OpenMP (Parallel MCTS)
*   **GUI:** PySide6 / PyQt6 (Card Editor)
*   **データ:** JSON (nlohmann/json)

---

## 2. コアエンジンアーキテクチャ (C++ Core Engine)

ゲームエンジンは `src/` 配下に配置され、以下の主要コンポーネントで構成されています。

### 2.1 ゲーム状態管理 (GameState)
`dm::core::GameState` はゲームの「ある瞬間のスナップショット」を保持します。探索速度を最大化するため、以下の設計がなされています。
*   **フラットなメモリ構造:** ポインタ参照を極力排除し、`std::vector` や固定長配列を使用。
*   **CardInstance:** 各カードの実体。`CardID`（静的データへの参照）と動的な状態（タップ状態、付与された効果など）を持ちます。
*   **Zone Management:** `Zone`（ゾーン）はカードインスタンスIDのリストとして管理され、移動はリスト間のID移動として表現されます。
*   **Loop Detection:** ゲーム状態のハッシュ値（`calculate_hash`）を用いて、無限ループ（千日手）を即座に検知します。

### 2.2 アクション生成と実行 (Action System)
ゲームの進行は「アクションの生成」と「解決」のサイクルで行われます。

*   **ActionGenerator:** 現在のフェーズや保留中の効果（PendingEffect）に基づき、プレイヤーが選択可能なすべての合法手（Legal Actions）を生成します。
    *   **Strategy Pattern:** `MainPhaseStrategy`, `AttackPhaseStrategy`, `PendingEffectStrategy` など、状況に応じた生成ロジックが分離されています。
*   **EffectResolver:** 選択されたアクションを受け取り、ゲーム状態を更新します。
    *   **Stack System:** プレイされたカードやトリガー能力は一度 `PendingEffect` としてスタック（`game_state.pending_effects`）に積まれ、LIFO（後入れ先出し）またはルールに基づく順序で処理されます。

### 2.3 汎用カードシステム (GenericCardSystem)
カードの効果処理はハードコードではなく、データ駆動型（Data-Driven）で設計されています。
*   **IActionHandler:** 「カードを引く」「破壊する」「マナ加速する」といった原子的操作は `IActionHandler` インターフェースを実装した各ハンドラ（`DrawHandler`, `DestroyHandler`等）に委譲されます。
*   **TargetUtils:** フィルタ条件（文明、コスト、種族など）に基づく対象選択ロジックを一元管理します。

### 2.4 特殊メカニクス実装
*   **革命チェンジ (Revolution Change):** 攻撃時の `TriggerType::ON_ATTACK_FROM_HAND` を検知し、条件を満たす場合に手札との入れ替えアクションを生成します。
*   **ハイパーエナジー (Hyper Energy):** コスト支払い時に `EffectActionType::COST_REFERENCE` アクションを発行し、バトルゾーンのクリーチャーをタップすることでコストを軽減します。
*   **シールド・トリガー (Shield Trigger):** シールドブレイク時にトリガーフラグを確認し、`USE_SHIELD_TRIGGER` アクションを生成して割り込み処理を行います。

---

## 3. AI・学習パイプライン (AI & Training Pipeline)

AIは「AlphaZero」スタイルの強化学習アプローチを採用しています。

### 3.1 データ収集 (Data Collection)
`src/ai/data_collector.cpp` (および Pythonラッパー) が自己対戦を行います。
*   **ParallelRunner:** OpenMPを使用し、複数のゲームスレッド並列で対戦を実行。
*   **MCTS (Monte Carlo Tree Search):** 探索を行い、訪問回数分布（Policy）と勝率予測（Value）を教師データとして生成します。
*   **Batch Inference:** C++側で貯めた盤面状態をまとめてPython側に送り、ニューラルネットで一括推論することで、言語間のオーバーヘッドを最小化しています。

### 3.2 学習 (Training)
`python/training/train_simple.py` が収集されたデータを学習します。
*   **Input:** 盤面の特徴量ベクトル（各ゾーンのカード分布、手札、マナなど）。
*   **Output:** 方策（Policy: どのアクションを取るべきか）と価値（Value: 現在の勝率）。

### 3.3 推論と評価 (Inference & Evaluation)
*   **LethalSolver:** 詰み（リーサル）があるかを判定する専用ソルバー。単純な打点計算だけでなく、ブロッカーやトリガーの可能性も考慮（または近似）します。
*   **BeamSearchEvaluator:** 探索幅を制限したビームサーチにより、高速に数手先を読みます。

---

## 4. データ構造とツール (Data & Tools)

### 4.1 カードデータ (JSON Architecture)
カード定義は `data/cards.json` に集約され、`JsonLoader` によってロードされます。
*   **CardDefinition:** 名前、コスト、文明などの基本情報に加え、`effects` リストを持ちます。
*   **Effect & Action:** 効果は「条件（Condition）」と「アクション（Action）」のツリー構造で定義され、柔軟なカード作成が可能です。
    *   例: 「登場時（Condition: ON_PLAY）」に「相手クリーチャーを1体破壊（Action: DESTROY, Filter: OPPONENT_CREATURE）」。

### 4.2 カードエディタ (Card Editor Ver 2.0)
`python/gui/card_editor.py` で実装されたGUIツールです。
*   **機能:** JSONを直接編集することなく、GUI上でカードの能力やトリガーを作成・編集できます。
*   **検証:** 定義したロジックがエンジンの仕様に適合しているかを検証する機能を持ちます。

### 4.3 シナリオエディタ
特定の盤面状況（Scenario）を作成し、AIの特定状況下での挙動テストや、詰将棋的なパズル作成に使用します。

---

## 5. 開発・検証フロー

*   **Pythonバインディング:** `dm_ai_module` をインポートすることで、Pythonの `pytest` からC++の内部ロジックを直接テスト可能です（`add_card_to_hand` などで盤面を強制操作してテスト）。
*   **Verification Scripts:** `verify_performance.py` や `verify_lethal_puzzle.py` など、特定の機能群やパフォーマンスを検証する専用スクリプトが整備されています。

このアーキテクチャにより、ルールの正確性（C++）と開発の柔軟性（Python/JSON）を両立させています。

```
