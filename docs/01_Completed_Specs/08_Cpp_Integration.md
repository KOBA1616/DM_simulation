# 8. C++統合提案 (C++ Integration Proposal)

## 8.1 現状の統合状況
- **Core Logic**: 完全にC++化済み (`dm::engine`).
- **Search**: MCTSをC++に移植し、Pythonオーバーヘッドを排除。
- **Evaluation**: 単純なヒューリスティック評価はC++で完結。
- **Keyword Support**: `SLAYER`, `POWER_ATTACKER`, `BREAKER` 系のロジックを `EffectResolver` に実装済み。

## 8.2 今後の統合ロードマップ
1.  **Neural Network Inference in C++ (LibTorch)**:
    - **LibTorch Integration**: 行動決定（Actor）と盤面評価（Evaluator）の両方をC++側のLibTorchで実装する。
    - これにより、Python <-> C++ 間の通信コストを完全になくし、推論速度を最大化する。
2.  **Hardware Acceleration (Quantization)**:
    - **Int8 Quantization**: 学習済みモデルをC++側でロードする際にInt8量子化を適用。
    - 最近のCPU（Intel AVX-512/VNNI, ARM NEON）の低精度演算命令を活用し、推論レイテンシを大幅に削減する。
3.  **Full C++ Training Loop (Hybrid Architecture)**:
    - **Concept**: "Python Brain, C++ Muscle"
    - **Python (Brain)**: 学習（Backprop）、オプティマイザ（AdamW等）、ハイパーパラメータ管理、モデル保存を担当。PyTorchの柔軟性を活かす。
    - **C++ (Muscle)**: 推論（Forward）、自己対戦（Self-Play）、MCTS探索、ログ生成を担当。LibTorchとゼロコピー転送により極限まで高速化する。
    - **Workflow**:
        1. Python: モデルの重みを共有メモリまたはファイル経由でC++に渡す。
        2. C++: 高速に数千試合の自己対戦を行い、学習データ（State, Policy, Value）を生成する。
        3. Python: 生成されたデータを読み込み、GPUで学習を行い、重みを更新する。
    - これにより、C++での複雑な勾配計算実装を回避しつつ、実行時間の99%を占める「対戦」部分を高速化する。
