Summary
-------
Fix indentation and pytest conversion for `tests/test_phase4_e2e.py` and ensure `tests/test_action_to_command.py` expectations align with Phase4 normalization.

Changes
-------
- Updated `tests/test_phase4_e2e.py`:
  - Replaced unittest-style `self` assertions with pytest `assert`.
  - Replaced undefined `Action` usage with `MockAction` defined in the test file.
  - Fixed indentation, variable scoping, and corrected use of `card_db` / `player` references.
  - Ensures commands execute via `EngineCompat.ExecuteCommand` and performs basic post-condition checks.

- No changes to engine bindings; tests now run successfully locally.

Tests
-----
Run locally:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q tests/test_phase4_e2e.py tests/test_action_to_command.py -q
```

Both tests pass on the development environment used to prepare this PR.

Notes / Next Steps
------------------
- Push branch and open PR for CI verification.
- If CI uses a different Python/C++ runtime, ensure compiled extension and runtime DLLs are available in `bin/` in CI environment or adjust `tests/conftest.py` import shims accordingly.
- Optional: expand `test_phase4_e2e` to cover more turn sequences and assert engine-side events/logs.
