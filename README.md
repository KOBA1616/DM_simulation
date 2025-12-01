# DM_simulation

Duel Masters Trading Card Game Simulator with AI.

## Project Structure

```text
dm_simulation/
├── bin/                        # Compiled Executables
├── build/                      # CMake Build Directory
├── config/                     # Runtime Configs (rules.json, train_config.yaml)
├── data/                       # Assets (Cards, Decks, Logs)
├── docs/                       # Documentation
├── models/                     # Trained AI Models (.pth)
├── python/                     # Python Source Code
│   ├── gui/                    # PyQt Frontend
│   ├── py_ai/                  # Python AI Modules
│   ├── scripts/                # Utility Scripts (train.py, validator.py)
│   └── tests/                  # Python Unit Tests
├── scripts/                    # Shell/PowerShell Helper Scripts
├── src/                        # C++ Source Code
│   ├── ai/                     # AI Components (MCTS, Evaluator)
│   ├── core/                   # Core Types & Constants
│   ├── engine/                 # Game Engine Logic
│   └── python/                 # Pybind11 Bindings
├── tests/                      # Integration Tests
└── tools/                      # Development Tools
```

## Getting Started

See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for setup and development instructions.

## Documentation

- [Project Overview](docs/01_Project_Overview.md)
- [System Architecture](docs/02_System_Architecture.md)
- [Detailed Implementation Steps](docs/11_Detailed_Implementation_Steps.md)
