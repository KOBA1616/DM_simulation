# Duel Masters AI Simulator - Ultimate Master Specification (v2.2)

## 1. プロジェクト概要 (Project Overview)
- **プロジェクト名**: DM_AI_Simulator
- **目的**: デュエル・マスターズのメタゲーム解明（最強デッキ探索）およびプレイングの最適化。
- **開発哲学**:
    - **High-Speed**: C++20によるビットボード・メモリプール・ゼロコピー転送を駆使した超高速エンジン。
    - **AlphaZero-based**: MCTS (C++) + MLPによる自己対戦強化学習。
    - **Evolutionary**: 評価ベースのサイドボード交換によるデッキ進化。
    - **Scientific**: PyQt6による可視化とヒートマップ分析。
    - **Version Control**: 各開発ステップ完了ごとにGitコミットを行い、進捗を確実に保存する。
    - **Quality Control**: 各開発フェーズ終了時に静的解析を実行し、コード品質を維持する。

## 2. システムアーキテクチャ (System Architecture)

### 2.1 技術スタック
- **Core Engine**: C++20 (GCC/Clang/MSVC) - std::span, concepts, modules (optional)
- **Binding**: Pybind11 (Zero-copy tensor passing)
- **AI/ML**: Python 3.10+, PyTorch (CUDA/MPS auto-detect)
- **Search**: C++ MCTS (Multi-threaded, Batch Evaluation) [Updated]
- **GUI**: PyQt6 (Non-blocking, Polling-based update)
- **Build**: CMake 3.14+
- **Test**: Python pytest for logic, validator.py for CSV integrity

### 2.2 ディレクトリ構成 & 名前空間
名前空間はディレクトリ階層に準拠する（例: `dm::engine::ActionGenerator`）。

```text
dm_simulation/
├── CMakeLists.txt              # Build Config
├── config/                     # Runtime Configs (rules.json, train_config.yaml)
├── data/                       # Assets
│   ├── cards.csv               # Card DB
│   ├── card_effects.json       # Card Effect Definitions (for Generator)
│   ├── decks/                  # Deck Files
│   └── logs/                   # Debug Logs
├── docs/                       # Documentation
│   └── DM_AI_Master_Spec_Final.md
├── src/                        # C++ Source (Namespace Hierarchy)
│   ├── core/                   # [dm::core] Dependencies-free
│   ├── engine/                 # [dm::engine] Game Logic
│   │   ├── action_gen/         # ActionGenerator
│   │   ├── flow/               # PhaseManager
│   │   ├── effects/            # EffectResolver, GeneratedEffects
│   │   └── mana/               # ManaSystem
│   ├── ai/                     # [dm::ai] AI Components
│   │   ├── mcts/               # C++ MCTS Implementation [New]
│   │   ├── evaluator/          # Heuristic Evaluator [New]
│   │   └── encoders/           # TensorConverter
│   ├── utils/                  # [dm::utils] RNG, CSV Loader
│   └── python/                 # Pybind11 Interface
├── python/                     # Python Source Code [Refactored]
│   ├── gui/                    # PyQt Frontend
│   │   ├── app.py              # Main Window
│   │   └── widgets/            # Custom Widgets (GraphView, DetailPanel)
│   ├── py_ai/                  # Python AI Modules (Training, GA)
│   └── scripts/                # Entry Points (train.py, test_ai.py)
├── tests/                      # Integration Tests
│   ├── test_card_creation_integration.py
│   └── test_spiral_gate.py
├── tools/                      # Development Tools
│   └── card_gen/               # Card Logic Generator (JSON -> C++) [New]
├── models/                     # Trained Models
└── bin/                        # Compiled Executables
```

## 3. コア・データ仕様 (Core Data Specs)

### 3.1 定数と制限 (constants.hpp)
- **MAX_HAND_SIZE**: 20
- **MAX_BATTLE_SIZE**: 20
- **MAX_MANA_SIZE**: 20
- **MAX_GRAVE_SEARCH**: 20
- **TURN_LIMIT**: 100
- **POWER_INFINITY**: 32000

### 3.2 カードデータ構造 (card_def.hpp)
- **ID管理**: CardID (uint16_t).
- **Keywords**:
    - **Basic**: `SPEED_ATTACKER`, `BLOCKER`, `SLAYER`, `EVOLUTION`.
    - **Breakers**: `DOUBLE_BREAKER`, `TRIPLE_BREAKER`.
    - **Parameterized**: `POWER_ATTACKER` (Bonus Value).
- **Filter Parsing**: CSVロード時に文字列条件をID化して保持.
- **Mode Selection**: ModalEffectGroup 構造体による複数選択管理.

### 3.3 盤面状態 (game_state.hpp)
- **Determinism**: `std::mt19937` をState内に保持し、シード値による完全再現を保証.
- **Incomplete Info**: 学習用Viewでは相手の手札・山札の中身をマスクまたはランダム化.
- **Error Handling**: 異常状態で `std::runtime_error` を送出し、即時停止.

## 4. ゲームルール詳細 (Detailed Game Rules)
(Ver 2.0と同様のため省略。変更なし)

## 5. AI & 機械学習仕様 (AI/ML Specs)

### 5.1 ネットワーク構造
- **Backbone**: MLP (多層パーセプトロン) 5層。
- **Flow**: Input (Tensor) -> FC(1024) -> ReLU -> ... -> FC(ActionSize).
- **Update Policy**: 勝率閾値 (Threshold) 更新。

### 5.2 入出力マッピング
- **Input Tensor**: Zero-Copy (C++ -> Python).
- **Action Space**: Flattened Fixed Vector (approx. 600 dim).

### 5.3 MCTS (Monte Carlo Tree Search) [Updated]
- **Implementation**: C++ (`src/ai/mcts/`) による高速実装。
- **Evaluator**:
    - **Heuristic**: ルールベースの高速評価関数 (`HeuristicEvaluator`). GUI/Debug用。
    - **Neural**: PyTorchモデルによる推論 (Batching対応予定).
- **Determinization**: 未公開情報をランダム固定してプレイアウト (C++側で処理).
- **Performance**: Python版と比較して約100倍の探索速度を実現。

### 5.4 学習ループ
- **Replay Buffer**: Hybrid Buffer (Sliding Window + Golden Games).
- **Seed Mgmt**: 学習マネージャーによるシード配布.

## 6. メタゲーム進化 (Meta-Game Evolution)
(Ver 2.0と同様)

## 7. GUI & 開発ツール (Frontend/DevOps)

### 7.1 PyQt6 GUI [Updated]
- **Control**: クリックによるステップ実行.
- **Visualization**:
    - **MCTS Graph View**: 探索木をグラフィカルに表示し、AIの思考プロセスを可視化 [New].
    - **Card Detail Panel**: ホバー時にカードの詳細スペックを表示 [New].
    - **God View**: デバッグ用に相手の手札・シールドを透視するモード [New].
- **Concurrency**: ポーリング方式による非同期更新.

### 7.2 開発補助
- **Card Generator**: `tools/card_gen/`
    - JSON定義ファイル (`data/card_effects.json`) からC++のカード効果実装コード (`generated_effects.hpp`) を自動生成。
    - **Supported Effects**: `mana_charge`, `draw_card`, `tap_all`, `destroy`, `bounce`.
    - 単純な効果（ドロー、マナブースト、破壊、バウンス）の実装工数を大幅に削減。
- **Deck Builder**: GUI内蔵エディタ.

## 8. C++統合提案 (C++ Integration Proposal)

### 8.1 現状の統合状況
- **Core Logic**: 完全にC++化済み (`dm::engine`).
- **Search**: MCTSをC++に移植し、Pythonオーバーヘッドを排除。
- **Evaluation**: 単純なヒューリスティック評価はC++で完結。
- **Keyword Support**: `SLAYER`, `POWER_ATTACKER`, `BREAKER` 系のロジックを `EffectResolver` に実装済み。

### 8.2 今後の統合ロードマップ
1.  **Neural Network Inference in C++**:
    - 現在はPython側で推論しているが、`LibTorch` または `ONNX Runtime` を用いてC++側で推論を行うことで、Python <-> C++ 間の通信コストをゼロにする。
    - これにより、Self-Playの速度をさらに向上させる。
2.  **Full C++ Training Loop**:
    - 学習ループ（Self-Play -> Buffer -> Train）のうち、Self-Play部分を完全にC++バイナリとして独立させる。
    - Pythonは学習（Backprop）とモデル管理のみを担当する構成へ移行。

---
(以下、ルール詳細はVer 2.0と同様)
