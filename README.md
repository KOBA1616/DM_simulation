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

Preferred build system: CMake (use Ninja or your preferred generator). Avoid using Visual Studio solution files or MSVC project files checked into the repository — this project standardizes on CMake.

Quick start (Windows, recommended with Ninja):

```powershell
# Configure (use Ninja if available)
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
# Build
cmake --build build --config Release -- -j
# Run tests
C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe -m pytest -q
```

If you must use Visual Studio to inspect the native code, prefer generating project files from CMake instead of checking generated `.sln`/`.vcxproj` files into source control:

```powershell
# Generate Visual Studio solution (only for interactive debugging, do not commit generated files)
cmake -S . -B build-vs -G "Visual Studio 17 2022" -A x64
```

To remove legacy Visual Studio / MSVC artifacts from your working tree, use `scripts\clean_msvc.ps1` (it will prompt before deleting).

See [docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md) for more setup and development instructions.

## Documentation

- [Project Overview](docs/00_Overview/01_Project_Overview.md)
- [System Architecture](docs/00_Overview/02_System_Architecture.md)
- [Detailed Implementation Steps](docs/02_Planned_Specs/11_Detailed_Implementation_Steps.md)
- [API / Action→Command スキーマ](docs/api/action_command_mapping.md)
- [Command Pipeline Migration (フェーズ2)](docs/02_Planned_Specs/command_pipeline_migration.md)
- Notes and PR summaries: [docs/notes/](docs/notes/)
