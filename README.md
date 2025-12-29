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
├── dm_toolkit/                 # Python Toolkit & Modules
│   ├── ai/                     # Python AI Components
│   ├── gui/                    # PyQt Frontend
│   ├── training/               # Training Scripts & Modules
│   └── utils/                  # Utility Modules
├── scripts/                    # Shell/PowerShell Helper Scripts
├── src/                        # C++ Source Code
│   ├── ai/                     # AI Components (MCTS, Evaluator)
│   ├── bindings/               # Pybind11 Bindings
│   ├── core/                   # Core Types & Constants
│   └── engine/                 # Game Engine Logic
├── tests/                      # Python Unit Tests & Integration Tests
└── scripts/                    # Scripts & Development Tools
```

## Getting Started

See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for setup and development instructions.

## Documentation

- [Project Overview](docs/00_Overview/01_Project_Overview.md)
- [System Architecture](docs/00_Overview/02_System_Architecture.md)
- [Detailed Implementation Steps](docs/02_Planned_Specs/11_Detailed_Implementation_Steps.md)
- [API / Action→Command スキーマ](docs/api/action_command_mapping.md)
- [Command Pipeline Migration (フェーズ2)](docs/02_Planned_Specs/command_pipeline_migration.md)
- Notes and PR summaries: [docs/notes/](docs/notes/)
