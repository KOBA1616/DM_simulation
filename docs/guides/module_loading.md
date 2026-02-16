# Module Loading and Path Manipulation

This document lists the sources of module loading and `sys.path` manipulation in the codebase.

## 1. Module Loading Sources

### `dm_ai_module.py` (Root)
*   **Purpose**: Acts as the canonical import target for the C++ native extension.
*   **Mechanism**:
    *   Searches for the compiled `dm_ai_module` binary (`.pyd` or `.so`) in `bin/`, `build/`, and `release/` directories.
    *   Prioritizes local build artifacts over system-installed versions.
    *   Provides a pure Python stub fallback if the native module is missing (for lightweight tests/linting).

### `sitecustomize.py` (Root)
*   **Purpose**: Ensures the repository root is automatically added to `sys.path` when Python starts.
*   **Mechanism**:
    *   Inserts the directory containing `sitecustomize.py` (repo root) into `sys.path`.
    *   Adds `bin/` and `build/` directories to `sys.path` as a backup for finding compiled extensions.
    *   Attempts to clean up stale `__pycache__` directories.

### `scripts/run_gui.ps1`
*   **Purpose**: Launch script for the GUI on Windows.
*   **Mechanism**:
    *   Sets `PYTHONPATH` to include the project root (`$projectRoot;$env:PYTHONPATH`).
    *   Ensures `dm_toolkit` is importable.

### `python/gui/app.py`
*   **Purpose**: Wrapper script to launch the main application.
*   **Mechanism**:
    *   Delegates directly to `dm_toolkit.gui.app.main`.
    *   Relies on `sitecustomize.py` or environment variables for path setup.

### `python/gui/card_editor.py`
*   **Purpose**: Wrapper script to launch the Card Editor.
*   **Mechanism**:
    *   Explicitly appends the project root (`../../`) to `sys.path` if not present.
    *   Delegates to `dm_toolkit.gui.card_editor.main`.

## 2. Best Practices

*   **Do not duplicate path manipulation**: Rely on `sitecustomize.py` or the launcher scripts (`run_gui.ps1`) to set up the environment.
*   **Import `dm_ai_module` safely**: Always use `try-except ImportError` blocks if your script might run in an environment without the native module (e.g., CI, linting), or use the `dm_ai_module.py` stub capabilities.
