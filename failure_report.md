# Failure Report (extracted from full_test_output.log)

Summary: 20 failing tests observed after running full test suite. Prioritized fix order and short notes below.

## Failures (short list)
- python/tests/test_logic_actions.py::test_if_else_action — unexpected hand counts after resolve_action (draw logic mismatch).
- python/tests/test_logic_actions.py::test_if_implicit_filter_condition — IF filter not triggering draw (filter/resolve logic).
- python/tests/unit/commands/test_command_unification.py::test_action_conversion_to_command — AttributeError: `dm_ai_module` missing `PlayerIntent`.
- python/tests/unit/converter/test_action_to_command.py::TestActionToCommand::test_determinism_execution — draw command did not change hand (engine command execution mapping/CommandDef shape issue).
- python/tests/unit/test_architecture_refactor.py::TestTriggerManager::test_data_collector_constructors — `DataCollector()` takes no arguments (binding constructor mismatch).
- python/tests/unit/test_event_dispatch.py::TestEventDispatch::test_zone_enter_trigger — Trigger not queued upon entering battle zone (event dispatch / pending effects).
- python/tests/unit/test_game_instance_wrapper.py::test_game_instance_wrapper — PASS action did not advance phase (phase manager / resolve_action behavior).
- python/tests/unit/test_modifier_pipeline.py::TestModifierPipeline::test_compile_action_and_execute — AttributeError: `EffectActionType` missing `APPLY_MODIFIER` (binding missing enum member).
- python/tests/unit/test_restriction_system.py::* — AttributeError: missing `command_history` or `PassiveType` members (API surface differences).
- python/tests/unit/test_self_attention.py::* — `SelfAttention.initialize_weights` missing (binding incomplete).
- python/tests/unit/test_tokenization.py::test_token_encoding_basic — Tokenizer output mismatch (tokens[0] == 0, expected 1).
- python/tests/verification/test_legacy_conversion.py::test_verify_legacy_conversion — Many primitives produced "No Effect Loaded" (conversion/mapping registry gaps).
- python/tests/verification/test_phase1_commands.py::* — RETURN_TO_HAND / TAP / UNTAP commands not modifying state as expected (CommandSystem execution semantics / filtering failing).

## Prioritized recommended fix order
1. Binding/API surface fixes (highest priority)
   - Add or expose missing enums/types in `dm_ai_module` (`PlayerIntent`, `PassiveType` variants, `EffectActionType` members, `CommandType` shape if needed).
   - Restore missing methods/constructors: `DataCollector` constructors, `SelfAttention.initialize_weights`.
   - Ensure `CommandDef` exposed attributes match Python expectations (e.g., `instance_id`).
   Rationale: Many failures stem from missing attributes or incompatible bindings; fixing these will unblock many tests.

2. CommandDef/CommandSystem shape and mapping
   - Fix attribute mismatches (`instance_id`) and mapping code in `dm_toolkit.engine.compat` that expects certain fields.
   - Ensure `map_action` and `EngineCompat.ExecuteCommand` produce/accept the expected command dict/CommandDef shapes.

3. Engine semantics and event/phase behaviors
   - Investigate why PASS action didn't change phase and why triggers aren't queued; inspect `GameInstance`, `PhaseManager`, and pending effect dispatch.
   - Verify `resolve_action` and `CommandSystem.execute_command` apply state changes (draw/tap/untap/return_to_hand) as expected.

4. Legacy conversion and tokenization
   - Rebuild/inspect action-to-effect/conversion registry to populate missing legacy conversions.
   - Fix TokenConverter or encoding params causing tokens[0] mismatch.

5. Tests & validation
   - After each fix, run targeted pytest for affected tests; update `full_test_output.log` and iterate.

## Immediate next action (I will start now)
- Implement compatibility shims in Python for missing binding members (step 1). I will open `dm_ai_module.py` and `dm_toolkit/engine/compat.py` to identify quick shims that can restore expected attributes without changing C++ code, then run targeted tests.

---

If this plan is acceptable, I'll proceed to inspect `dm_ai_module.py` now and add minimal shims for missing enums/members.