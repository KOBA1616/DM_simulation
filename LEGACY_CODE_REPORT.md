# Legacy and Deprecated Code Report

This report documents the findings regarding legacy code, deprecated features, and cleanup actions taken.

## 1. Duplicate Method in `dm_toolkit/engine/compat.py` (Fixed)

**Issue:** The class `EngineCompat` contained two definitions of `ActionGenerator_generate_legal_commands`.
1.  The first definition raised a `RuntimeError` ("deprecated").
2.  The second definition (immediately following) delegated to `dm_toolkit.commands_v2`.

**Impact:** Because Python overwrites methods with the same name, the first definition was dead code. The intended "deprecation error" was never triggered.

**Action Taken:** The first definition has been removed. The method now correctly delegates to the new command system, maintaining backward compatibility for the GUI and other tools.

## 2. `dm_ai_module.py` Status

**Finding:** `dm_ai_module.py` serves as a critical hybrid shim.
- It attempts to load the native C++ extension (`dm_ai_module.so` / `.pyd`).
- If successful, it injects additional Python helper classes into the native module.
- If unsuccessful (e.g., in lightweight CI or dev environments without a build), it provides a "pure Python" fallback implementation of the game engine.

**Recommendation:** Do **not** delete this file. It is essential for cross-platform compatibility and testing.

## 3. `ReactionWindow` Discrepancy

**Finding:**
- Documentation/Memory suggested that `ReactionWindow` had been removed.
- **Reality:** The file `src/engine/systems/effects/reaction_window.hpp` still exists and is included by `src/core/game_state.hpp`.

**Recommendation:** Do not delete `ReactionWindow` files without a comprehensive C++ refactor, as they are likely still compiled and linked.

## 4. "Action" vs "Command" Deprecation

**Finding:** The codebase is migrating from "Actions" (legacy dicts) to "Commands" (structured objects).
- `dm_toolkit/unified_execution.py` contains active warnings: `"Passing Action-like objects to unified execution is deprecated"`.
- `dm_toolkit/action_mapper.py` has already been removed (good).

**Recommendation:** Continue to heed warnings in `unified_execution.py`. Use `dm_toolkit.commands_v2` for generating commands in new code.

## 5. Scripts and Tools

**Finding:** The `scripts/` directory appears clean. Utilities like `check_native_symbols.py` and `import_dm_ai.py` are functional and should be kept.

## 6. Conclusion

The codebase is generally clean, with deprecated items clearly marked or removed. Future cleanups should focus on the C++ side (`ReactionWindow`) once the architecture permits.
