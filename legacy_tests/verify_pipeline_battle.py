import sys
import os

# Add the bin directory to sys.path
bin_path = os.path.join(os.getcwd(), 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module. Make sure the module is built and in the bin directory.")
    sys.exit(1)

def verify_pipeline_battle():
    print("Verifying Pipeline RESOLVE_BATTLE...")

    # 1. Setup GameState
    state = dm_ai_module.GameState(100)
    state.active_player_id = 0

    # 2. Setup Cards (Attacker and Target)
    # Attacker (P0): Power 3000
    state.add_test_card_to_battle(0, 100, 10, True, False) # ID=100, Inst=10, Tapped, NotSick
    # Target (P1): Power 2000
    state.add_test_card_to_battle(1, 101, 20, False, False) # ID=101, Inst=20

    # 3. Define Card Data
    card_db = {
        100: dm_ai_module.CardDefinition(100, "Attacker", "FIRE", [], 3, 3000, dm_ai_module.CardKeywords(), []),
        101: dm_ai_module.CardDefinition(101, "Target", "WATER", [], 2, 2000, dm_ai_module.CardKeywords(), [])
    }

    # 4. Invoke RESOLVE_BATTLE action
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.RESOLVE_BATTLE
    # Currently implementation seems to assume attacker is source? Or we need to see how RESOLVE_BATTLE is handled.
    # Instruction args: source (attacker), target (defender) - from where?
    # ActionType doesn't pass args directly, GameLogicSystem::dispatch_action might map it.
    # Looking at dispatch_action in GameLogicSystem.cpp:
    # case ActionType::RESOLVE_BATTLE:
    #    GameLogicSystem::handle_resolve_battle(...)

    # But dispatch_action for RESOLVE_BATTLE was NOT implemented in the snippet I read!
    # "case ActionType::RESOLVE_BATTLE:" was NOT present in the switch.

    # So this test is expected to fail or do nothing if I run it now.

    # I'll check if it does anything.

    # But I can't pass args via Action if dispatch_action ignores them.
    # I might need to construct Instruction directly if I could, but Python binding for PipelineExecutor allows execute(instructions...)

    # Let's try executing via PipelineExecutor directly.
    inst = dm_ai_module.Instruction()
    inst.op = dm_ai_module.InstructionOp.GAME_ACTION
    # We can't easily set JSON args from Python directly unless we use a wrapper or if binding supports it.
    # The binding for Instruction doesn't expose `args` setter for JSON.

    # So I have to rely on GameLogicSystem.resolve_action dispatching correctly.
    # If dispatch_action doesn't handle RESOLVE_BATTLE, I need to add it.

    # Let's try to run a dummy action and see if it crashes or does nothing.
    action.source_instance_id = 10
    action.target_instance_id = 20

    print("Executing Action...")
    dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

    # 5. Check results
    # Expectation: Target (Inst 20) should be in Graveyard (P1). Attacker (Inst 10) stays in Battle.

    p1_grave = state.get_zone(1, dm_ai_module.Zone.GRAVEYARD)
    p1_battle = state.get_zone(1, dm_ai_module.Zone.BATTLE)

    found_in_grave = 20 in p1_grave
    found_in_battle = 20 in p1_battle

    print(f"Target in Graveyard: {found_in_grave}")
    print(f"Target in Battle: {found_in_battle}")

    if found_in_grave and not found_in_battle:
        print("SUCCESS: Battle resolved correctly.")
    else:
        print("FAILURE: Battle did not resolve correctly.")

if __name__ == "__main__":
    verify_pipeline_battle()
