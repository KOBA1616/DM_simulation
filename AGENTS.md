# Development Policy and Architecture Guidelines

## 1. Command Normalization Policy
*   **Unified Entry Point:** All Action-to-Command conversions must go through `dm_toolkit.action_to_command.action_to_command`.
*   **Standardization:** Ensure `action_to_command.py` is the single source of truth for mapping legacy Action dictionaries to strict `GameCommand` structures.
*   **Goal:** Eliminate ad-hoc dictionary manipulation in test code and engine wrappers.

## 2. Compatibility and Post-Processing
*   **Minimize Dispersion:** Logic for backward compatibility and command post-processing should be centralized.
    *   `dm_toolkit.compat_wrappers`: For wrappers handling legacy API signatures.
    *   `dm_toolkit.unified_execution`: For executing standardized commands against the engine.
    *   `dm_toolkit.action_mapper`: For specific mapping logic (helper for `action_to_command`).
*   **Refactoring:** Avoid spreading "if legacy_mode:" checks throughout the codebase. encapsulate them in these modules.

## 3. Headless Testing and Stubbing
*   **Official Stubbing Route:** Use the formalized stub injection mechanism for headless environments (CI/No-GUI).
*   **Execution:** The standard way to run tests in headless mode is via `run_pytest_with_pyqt_stub.py`.
*   **Verification:** Ensure `python/tests/gui/test_gui_stubbing.py` passes to verify the stubbing infrastructure works before running the full suite.
*   **Pytest:** Invoke pytest through the unified runner to ensure consistent environment setup.

## 4. `dm_ai_module` Loading Strategy
*   **Source Loader Priority:** The Python wrapper (`dm_ai_module.py`) should prioritize loading the extension from the build directory (`build/`, `bin/`) over system-installed versions.
*   **Explicit Verification:** Only enforce native module presence when explicitly required by the test context (managed via `conftest.py`).
*   **Environment Setup:** Use `sitecustomize.py` or the module's own loader to ensure `sys.path` is correctly configured to find the compiled extension.
