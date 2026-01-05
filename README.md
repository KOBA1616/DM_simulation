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
cmake -S . -B build-msvc -G "Ninja" -DCMAKE_BUILD_TYPE=Release
# Build
cmake --build build-msvc --config Release -- -j
# Run tests
C:/Users/ichirou/DM_simulation/.venv/Scripts/python.exe -m pytest -q
```

## Local CI (Windows)

CI相当のビルド/テスト/mypyをローカルでまとめて実行し、ログを `dumps/logs` に保存します。

- Run: `pwsh -File scripts/run_ci_local.ps1`
- Log dir: `dumps/logs/ci_local_YYYYMMDD_HHMMSS/`

必要に応じてスキップ可能です。

- Skip build: `pwsh -File scripts/run_ci_local.ps1 -SkipBuild -SkipCTest`
- Skip mypy: `pwsh -File scripts/run_ci_local.ps1 -SkipMypy`

## Text Encoding

このリポジトリのテキストファイル（Python/C++/PowerShell/JSON/YAML/Markdownなど）は **UTF-8** で記述します。

- PythonでファイルI/Oをする場合は、Windowsの既定コードページ(cp932)に依存しないよう `encoding='utf-8'` を明示してください。
- PowerShellは `.editorconfig` により `.ps1` を `utf-8-bom` としています（Windows PowerShell 5.1互換のため）。

### 外部ファイル（Shift-JIS/CP932）の扱い

外部から持ち込まれた JSON/CSV/テキスト等が Shift-JIS（cp932 / windows-31j）だった場合、**リポジトリにはUTF-8へ変換してから追加**してください（Shift-JISのままコミットしない）。

- 変換（PowerShell / pwsh推奨）

```powershell
# 例: cp932 -> UTF-8
$src = "path\\to\\input.txt"
$dst = "path\\to\\input.utf8.txt"
[System.IO.File]::WriteAllText($dst, [System.IO.File]::ReadAllText($src, [Text.Encoding]::GetEncoding(932)), [Text.Encoding]::UTF8)
```

- 変換後に、アプリ/スクリプト側の読み込みは基本 `encoding='utf-8'` のままにします。
	- 「Shift-JISも自動で読める」挙動は、意図しない文字化けや環境差分（Windowsロケール依存）の原因になりやすいため、原則入れません。

Workspace cleanup (optional):

```powershell
# Preview what would be deleted/moved
./scripts/clean_workspace.ps1 -CleanCaches -MoveRootLogs -DryRun
# Clean caches and move root logs into dumps/logs/workspace/<timestamp>
./scripts/clean_workspace.ps1 -CleanCaches -MoveRootLogs -Force
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
