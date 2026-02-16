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

### Clone distribution (Windows)

このリポジトリは「GitHubでクローンしてもらい、セットアップスクリプトで相手PCの環境を確認してから構築する」運用を想定できます。

```powershell
git clone <YOUR_REPO_URL>
cd DM_simulation
pwsh -File .\scripts\setup_clone_windows.ps1
```

#### ビルドツールが無い（C++をビルドできない）場合

このプロジェクトのフルGUI（対戦シミュレータ/シナリオ/AI表示など）は `dm_ai_module`（C++拡張）が必要です。相手PCにビルドツールが無い場合は次のどちらかになります。

- **Build Tools を入れてビルドする（推奨）**
	- Visual Studio 2022 Build Tools + C++（VCTools）を導入してから `setup_clone_windows.ps1` を再実行
	- スクリプト側で best-effort の自動導入も試せます: `pwsh -File .\scripts\setup_clone_windows.ps1 -InstallVSBuildTools`

- **GUI表示レビューだけ行う（ネイティブ不要の範囲に限定）**
	- カードエディタ単体はネイティブ無しで起動できます:
		- `pwsh -File .\scripts\setup_gui_review_windows.ps1`
		- 2回目以降の起動は: `pwsh -File .\scripts\run_gui_review.ps1`

セットアップが通ったら、必要に応じて次を実行します。

- GUI起動: `pwsh -File .\scripts\run_gui.ps1`
- CI相当まとめ実行: `pwsh -File .\scripts\run_ci_local.ps1`

Quick start (Windows, recommended with Ninja):

```powershell
# Configure (use Ninja if available)
cmake -S . -B build-msvc -G "Ninja" -DCMAKE_BUILD_TYPE=Release
# Build
cmake --build build-msvc --config Release -- -j
# Run tests
python -m pytest -q
```

## Unified CLI (dm-cli)

`dm-cli` is the main entry point for headless operations and validation.

```bash
# Run interactive console (headless REPL)
./dm-cli console

# Run headless simulation (batch games)
./dm-cli sim --games 100

# Validate card data
./dm-cli validate data/cards.json
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

See [docs/development/workflow.md](docs/development/workflow.md) for more setup and development instructions.

## Documentation

- [Project Overview](docs/project/overview.md)
- [System Architecture](docs/architecture/engine.md)
- [Detailed Implementation Steps](docs/project/status.md)
- [API / Action→Command スキーマ](docs/engine/infrastructure/commands/action_command_mapping.md)
- [Command Pipeline Migration (フェーズ2)](docs/engine/dev/command_pipeline_migration.md)
- Notes and PR summaries: [docs/engine/reference/](docs/engine/reference/)

## Repository cleanup

一部のドキュメントとスクリプトをルートから `docs/` および `scripts/` に移動しました。古い重複ファイルはルートから削除されています。

- ドキュメント（例）: `docs/Specs/AGENTS.md`, `archive/docs/MEGA_LAST_BURST_IMPLEMENTATION.md`
- スクリプト（例）: `scripts/diagnose_game_training.py`, `scripts/train_fast.py`

必要であれば元の配置に戻すか参照パスを調整してください。
