Headless main-window simulator

Usage examples:

- Run interactive REPL:

  ```powershell
  .\.venv\Scripts\Activate.ps1
  $env:PYTHONPATH = (Get-Location).Path
  python .\scripts\headless_mainwindow.py
  ```

- Auto-run 10 iterations and dump minimal state to JSON:

  ```powershell
  python .\scripts\headless_mainwindow.py --auto 10 --dump-json out.json --log-level DEBUG
  ```

- Quick smoke test (used by tests):

  ```powershell
  python .\scripts\headless_mainwindow.py --auto 5
  ```

What it provides:
- A REPL that mirrors many main-window actions (`hand`, `deck`, `legal`, `play`, `step`, `run`, `auto`).
- `--auto N` to run automated iterations and exit.
- `--dump-json FILE` to write a minimal snapshot of player hands/decks on exit.
- `--log-level` to control verbosity.

This tool is intended for debugging and CI smoke checks without importing Qt.
