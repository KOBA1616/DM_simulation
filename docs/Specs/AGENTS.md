```markdown
# Development Policy and Architecture Guidelines

## 1. Command Normalization Policy
*   **Unified Entry Point:** All Action-to-Command conversions must go through `dm_toolkit.action_to_command.action_to_command`.
*   **Standardization:** Ensure `action_to_command.py` is the single source of truth for mapping legacy Action dictionaries to strict `GameCommand` structures.
*   **Goal:** Eliminate ad-hoc dictionary manipulation in test code and engine wrappers.
*   **Status:** ✅ Phase 1-5 Complete (2026年1月22日). Phase 6 (Quality Assurance) in progress.

## 2. Compatibility and Post-Processing
*   **Minimize Dispersion:** Logic for backward compatibility and command post-processing should be centralized.
    *   `dm_toolkit.compat_wrappers`: For wrappers handling legacy API signatures.
    *   `dm_toolkit.unified_execution`: For executing standardized commands against the engine.
    *   `dm_toolkit.action_mapper`: (注) リポジトリ内に明示的な `action_mapper` モジュールは存在しません。
        実装は `dm_toolkit.action_to_command` に統合されているか、廃止されている可能性があります。
        ドキュメント上の古い参照のため、必要に応じて削除または移動を検討してください。
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

## 5. Quality Assurance Strategy (Phase 6)
*   **Test Coverage Target:** Maintain 98%+ test pass rate (Current: 98.3% as of 2026年1月22日).
*   **Critical Issues:** Focus on remaining test failures in GUI stubbing and text generation.
*   **Continuous Improvement:** Address technical debt incrementally while maintaining system stability.

---

## Review Log

- **Reviewed on:** 2026-01-19
- **Reviewer:** repository maintenance run
- **Summary:** ドキュメントはコマンド正規化、互換性ラッパー、ヘッドレステスト方針、ネイティブモジュール読み込み優先度、QA戦略が明確に記載されている。`dm_toolkit.action_to_command.action_to_command` を単一の正規化入り口とする方針は現行実装のリファクタリング目標として適切。
- **Action items:**
    - 実装側の差分確認が必要（`dm_toolkit` と `dm_ai_module.py` の実装が方針に沿っているか）。
    - GUIスタブ関連テストの最新失敗ログを集めて優先度付けすること。
    - 本ドキュメントは最新版として `docs/requirements/requirements_review_20260119.md` に要約と推奨アクションを追加済み。

## Changelog

- 2026-01-19: レビュー追記（非破壊）。今後のアクションは `docs/requirements/requirements_review_20260119.md` を参照。

```
