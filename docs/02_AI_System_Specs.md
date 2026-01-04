# AI System Specifications (要件定義書 02)

## 1. 全体構成 (Architecture Overview)

### 1.1 Hybrid Architecture
*   **C++ Core**: MCTS探索、Lethal Solver、並列対戦実行 (`ParallelRunner`)。
*   **Python/PyTorch**: Neural Network定義、学習ループ、ハイパーパラメータ管理。
*   **Interface**: `dm_ai_module` を介して相互運用。

## 2. 実装済みコンポーネント (Implemented Components)

### 2.1 AlphaZero System
*   **Network**: ResNet (MLP) ベースの `AlphaZeroNetwork`。
*   **MCTS**: C++実装の `MCTS` クラス。UCB1 + PUCTアルゴリズム。
*   **Training Loop**: `train_simple.py` による Self-Play -> Data Collection -> Training -> Verification サイクル。

### 2.2 Transformer Model (Phase 4)
*   **DuelTransformer**: `dm_toolkit.ai.agent.transformer_model.DuelTransformer` に実装済み。
    *   **Self-Attention**: 盤面シーケンス全体の依存関係を学習。
    *   **Synergy Matrix**: カード間のシナジーバイアスをAttention Maskに注入。
*   **Status**: クラス定義は完了。学習パイプラインへの統合待ち。

### 2.3 Inference System (Phase 2)
*   **Deck Inference**: 相手のマナ・墓地・プレイ履歴から、未公開の手札とデッキ構成を確率的に推定する `DeckInference` クラス (C++)。
*   **PIMC**: Perfect Information Monte Carlo 法を用いた不完全情報ゲームへの対応（基盤実装済み）。

### 2.4 Evolution Ecosystem (Phase 3)
*   **Deck Evolution**: 対戦結果に基づきデッキレシピを進化させる `evolution_ecosystem.py`。
*   **Meta-Game**: 複数のデッキアーキタイプ (`meta_decks.json`) を保持し、メタ環境の変遷をシミュレートする。

## 3. 今後の要件 (Requirements)

### 3.1 Transformer Training
*   **Task**: `train_simple.py` を拡張（または `train_transformer.py` を作成）し、`DuelTransformer` を用いた学習を可能にする。
*   **Data Format**: Token Sequence 形式へのデータ変換処理の実装。

### 3.2 PBT (Population Based Training)
*   **Task**: 多数のエージェント（モデル＋デッキのペア）を並列で対戦させ、勝率の低い個体を淘汰・変異させる完全なPBTパイプラインの構築。
