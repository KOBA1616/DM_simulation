
import sys
import os

# Add the bin directory to sys.path to allow importing dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure the C++ module is built and in the bin directory.")
    sys.exit(1)

def verify_query_command():
    print("Verifying QUERY Command Integration...")

    # 1. Initialize Game State
    state = dm_ai_module.GameState(100)

    # 2. Setup Cards
    player_id = 0
    # Add some targets (e.g. Battle Zone)
    dm_ai_module.GameState.add_test_card_to_battle(state, player_id, 1, 0, False, False)
    dm_ai_module.GameState.add_test_card_to_battle(state, 1, 1, 1, False, False)

    # Register dummy card data for ID 1
    c = dm_ai_module.CardData(1, "Dummy", 1, "FIRE", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c)

    # 3. Create Query Command
    cmd_query = dm_ai_module.CommandDef()
    cmd_query.type = dm_ai_module.JsonCommandType.QUERY
    cmd_query.str_param = "SELECT_TARGET"
    cmd_query.target_group = dm_ai_module.TargetScope.ALL_PLAYERS

    # Filter: Creatures in Battle Zone
    filter_creatures = dm_ai_module.FilterDef()
    filter_creatures.zones = ["BATTLE_ZONE"]
    cmd_query.target_filter = filter_creatures

    # 4. Execute Command
    print("Executing QUERY command...")
    context = {}

    # Check initial state
    print(f"Initial waiting_for_user_input: {state.waiting_for_user_input}")

    dm_ai_module.CommandSystem.execute_command_with_context(state, cmd_query, -1, player_id, context)

    # 5. Verify State
    print(f"Post-execution waiting_for_user_input: {state.waiting_for_user_input}")

    if not state.waiting_for_user_input:
        print("FAILURE: State should be waiting for user input.")
        return

    # Check pending query details
    if state.pending_query is None:
        print("FAILURE: pending_query is None.") # Might be exposed as None if optional binding is tricky
        # The binding for std::optional<QueryContext> might expose it directly or via checked access.
        # Let's see bindings.
    else:
        print(f"Pending Query ID: {state.pending_query.query_id}")
        print(f"Pending Query Type: {state.pending_query.query_type}")
        print(f"Valid Targets: {state.pending_query.valid_target_ids}")

        # We expect 2 targets (instance 0 and 1)
        valid_targets = state.pending_query.valid_target_ids
        if len(valid_targets) != 2:
             print(f"FAILURE: Expected 2 valid targets, got {len(valid_targets)}")
        else:
             print("SUCCESS: Query Command Verified!")

if __name__ == "__main__":
    verify_query_command()
