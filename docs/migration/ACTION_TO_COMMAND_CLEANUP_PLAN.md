# Action to Command Migration - Final Cleanup Plan

## Status
- **Current State**: Command-first migration is complete. All training scripts and tools use command-based APIs.
- **Test Status**: 69 passed, 4 skipped (all passing)
- **Next Phase**: Remove legacy Action-based code and deprecated files

## Cleanup Phases

### Phase 1: Identify Deprecated Files ✓
Files identified for removal:
1. **Test Files** (legacy verification, no longer needed):
   - `tests/verify_action_generator.py` - Tests old ActionGenerator (replaced by command generators)
   - `tests/verify_action_to_command.py` - Tests action-to-command mapping (compatibility layer)
   - `tests/verify_action_to_command_strict.py` - Strict parity tests (migration complete)
   - `tests/verify_buffer_actions.py` - Buffer action tests (covered by command tests)

2. **Diagnostic Scripts** (using old resolve_action API):
   - `scripts/diag_pending_actions.py` - Uses `game.resolve_action()` directly
   - `scripts/diag_spell_test.py` - Uses `game.resolve_action()` directly
   - `scripts/diag_hime_play.py` - Uses `game.resolve_action()` directly

3. **Legacy Test Files**:
   - `tests/test_no_direct_execute_action.py` - Policy enforcement (migration complete)

### Phase 2: Update Remaining Scripts ✓
Scripts still using `resolve_action` that need updating:
- `simple_play_test.py`
- `scripts/inspect_selfplay_state.py`
- `scripts/selfplay_long.py`
- `python/tests/verification/verify_meta_demo.py`

### Phase 3: Remove Compatibility Shims
After confirming all code uses command-first APIs:
1. Mark `dm_toolkit/action_to_command.py` as deprecated (keep for data migration only)
2. Mark `dm_toolkit/compat_wrappers.py` as deprecated
3. Remove ActionGenerator references from:
   - `dm_toolkit/unified_execution.py`
   - `dm_toolkit/gui/headless.py`
   - `dm_toolkit/gui/game_session.py`
   - `dm_toolkit/engine/compat.py`
   - `dm_toolkit/commands.py`

### Phase 4: Documentation Updates
1. Update README.md to reflect command-first architecture
2. Archive migration guides to `docs/migration/archive/`
3. Create final architecture documentation

## Implementation Steps

### Step 1: Remove Legacy Test Files
```powershell
# Remove old verification tests
Remove-Item tests/verify_action_generator.py
Remove-Item tests/verify_action_to_command.py
Remove-Item tests/verify_action_to_command_strict.py
Remove-Item tests/verify_buffer_actions.py
Remove-Item tests/test_no_direct_execute_action.py
```

### Step 2: Remove Diagnostic Scripts
```powershell
# Remove old diagnostic scripts
Remove-Item scripts/diag_pending_actions.py
Remove-Item scripts/diag_spell_test.py
Remove-Item scripts/diag_hime_play.py
```

### Step 3: Update Remaining Scripts
Convert `simple_play_test.py`, `scripts/inspect_selfplay_state.py`, `scripts/selfplay_long.py` to use command-first APIs.

### Step 4: Mark Compatibility Layers as Deprecated
Add deprecation warnings to:
- `dm_toolkit/action_to_command.py`
- `dm_toolkit/compat_wrappers.py`

### Step 5: Clean Up ActionGenerator References
Remove or deprecate ActionGenerator fallback code in:
- `dm_toolkit/unified_execution.py`
- `dm_toolkit/gui/headless.py`
- `dm_toolkit/gui/game_session.py`
- `dm_toolkit/engine/compat.py`
- `dm_toolkit/commands.py`

### Step 6: Run Full Test Suite
Verify all tests pass after cleanup:
```powershell
pytest tests/ -v
```

### Step 7: Update Documentation
- Update MIGRATION_ACTION_TO_COMMAND_GUIDE.md with completion status
- Create architecture documentation for command-first system
- Archive old migration docs

## Risks and Mitigation

### Risk 1: Hidden Dependencies
**Mitigation**: Run full test suite after each cleanup step. Use git to track changes and enable easy rollback.

### Risk 2: External Tools
**Mitigation**: Keep compatibility layers marked as deprecated but functional for 1-2 releases before removal.

### Risk 3: Data Migration
**Mitigation**: Keep `action_to_command.py` available for converting old training data, even after code migration is complete.

## Success Criteria
- [ ] All legacy test files removed
- [ ] All diagnostic scripts removed or updated
- [ ] All remaining scripts use command-first APIs
- [ ] Compatibility layers marked as deprecated
- [ ] ActionGenerator references removed from core code
- [ ] Full test suite passes (69+ tests)
- [ ] Documentation updated
- [ ] MIGRATION_ACTION_TO_COMMAND_GUIDE.md marked as complete

## Timeline
- Phase 1-2: Immediate (file removal)
- Phase 3: 1 day (script updates)
- Phase 4-5: 1 day (deprecation and cleanup)
- Phase 6-7: 1 day (testing and documentation)

**Total Estimated Time**: 2-3 days
