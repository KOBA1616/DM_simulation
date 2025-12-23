# 2. システムアーキテクチャ (System Architecture)

## 2.1 技術スタック
- **Core Engine**: C++20 (GCC/Clang/MSVC) - std::span, concepts, modules (optional)
- **Binding**: Pybind11 (Zero-copy tensor passing)
- **AI/ML**: Python 3.10+, PyTorch (CUDA/MPS auto-detect)
- **Search**: C++ MCTS (Multi-threaded, Batch Evaluation) [Updated]
- **GUI**: PyQt6 (Non-blocking, Polling-based update)
- **Build**: CMake 3.14+
- **Test**: Python pytest for logic, validator.py for CSV integrity

### 2.3 外部インターフェース (External Interfaces) [New]
- **Gymnasium / PettingZoo Compliant**:
    - AIとゲームエンジンの通信部分を、強化学習の標準規格である **OpenAI Gym (Gymnasium)** および **PettingZoo** (マルチエージェント用) に準拠させる。
    - これにより、**Ray/RLlib, Stable Baselines3, CleanRL** などの既存の強力な強化学習ライブラリを直接接続して利用可能にする。

## 2.2 ディレクトリ構成 & 名前空間
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
│   ├── DM_AI_Master_Spec_Final.md
│   ├── 01_Project_Overview.md
│   ├── 02_System_Architecture.md
│   └── ...
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
│   └── bindings/               # Pybind11 Interface
├── dm_toolkit/                 # Python Toolkit & Modules
│   ├── gui/                    # PyQt Frontend
│   │   ├── app.py              # Main Window
│   │   └── widgets/            # Custom Widgets (GraphView, DetailPanel)
│   ├── ai/                     # Python AI Modules
│   └── training/               # Training Scripts
├── tests/                      # Python Unit Tests & Integration Tests
│   ├── test_card_creation_integration.py
│   └── test_spiral_gate.py
├── scripts/                    # Scripts & Development Tools
│   └── python/                 # Python Utilities (Card Generator, etc.)
├── models/                     # Trained Models
└── bin/                        # Compiled Executables
```
