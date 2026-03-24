# PR Summary: Card Editor Refactor (TDD micro-tasks)

This PR collects several small, test-driven changes aimed at improving the card editor's robustness and maintainability.

High-level themes:
- Replace raw Qt `.connect(...)` calls with `safe_connect(...)` across `dm_toolkit/gui/widgets` and editor areas to make UI initialization robust in headless/test environments.
- Add typed parameter models for editor commands: `QueryParams`, `TransitionParams`, `ModifierParams` (Pydantic v2).
- Add detection tests to ensure no raw `.connect(` remains in targeted areas.
- Add a lightweight CIR cost analyzer and a unit test to gather metrics.

Files added or significantly changed (highlights):
- `dm_toolkit/gui/widgets/*` (many files): replaced `.connect` usages with `safe_connect`.
- `dm_toolkit/gui/editor/*` (selected files): detection tests and some replacements; `CardEditor` already uses dispatch; added `models/params.py`.
- `dm_toolkit/gui/editor/models/params.py`: `QueryParams`, `TransitionParams`, `ModifierParams`.
- `python/tests/unit/test_widgets_no_raw_connect.py` and `python/tests/unit/test_editor_no_raw_connect.py`: detection tests for `.connect(`.
- `python/tests/unit/test_params_models.py`: tests for the new param models.
- `tools/cir_cost.py` and `python/tests/unit/test_cir_cost_analysis.py`: CIR adoption metric tool and test.

Testing guidance (run locally):
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest python/tests/unit -q
python -m pytest python/tests/gui -q
```

Review notes / potential risks:
- `safe_connect` wrapper changes the call-site shape: calls now use `safe_connect(obj, 'signalName', slot)` for consistency. Review any dynamic signal names or unusual uses.
- The automated replacements were conservative but manual review is recommended for complex signal chains (e.g., `QAction` closures, model signals).

If you want I can split this PR into smaller commits (widgets first, editor second, models/tests third).
