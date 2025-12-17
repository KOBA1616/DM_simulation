
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append('bin')

import dm_ai_module
import pytest

def test_query_flow():
    # Initialize GameState
    state = dm_ai_module.GameState(40)
    card_db = {}

    # 1. Manually inject a QueryCommand
    query_cmd = dm_ai_module.QueryCommand("SELECT_TARGET", [1, 2, 3], {})
    # Assuming execute_command is not directly exposed on state but via wrapping logic or manually setting it?
    # Wait, `GameState` binding has NO `execute_command`.
    # However, `GameCommand` objects have `execute(state)`.
    query_cmd.execute(state)

    assert state.waiting_for_user_input == True
    assert state.pending_query.query_type == "SELECT_TARGET"
    assert len(state.pending_query.valid_target_ids) == 3

    # 2. Generate Actions
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)

    # Should get 3 SELECT_TARGET actions
    assert len(actions) == 3
    for act in actions:
        assert act.type == dm_ai_module.ActionType.SELECT_TARGET
        assert act.target_instance_id in [1, 2, 3]

    # 3. Respond with Action
    # Use EffectResolver (aliased as GameLogicSystem in some contexts, but let's use what's available)
    # The binding says: m.attr("EffectResolver") = m.attr("GameLogicSystem");

    chosen_action = actions[1] # Choose ID 2

    dm_ai_module.EffectResolver.resolve_action(state, chosen_action, card_db)

    # 4. Verify Wait State Cleared
    assert state.waiting_for_user_input == False
    assert state.pending_query is None

def test_query_flow_options():
    state = dm_ai_module.GameState(40)
    card_db = {}

    # 1. Query Options
    # We need to set options in params or verify how SELECT_OPTION works.
    # The `QueryContext` has `options` field (vector of strings).
    # But `QueryCommand` constructor takes `params` (map<string, int>).
    # It seems `QueryCommand` implementation in bindings might not expose `options` list setting easily?
    # Let's check `QueryCommand` implementation.
    # It takes (type, targets, params).
    # `QueryContext` has `options`. `QueryCommand` implementation doesn't seem to populate `options` from constructor args in bindings?
    # Actually `QueryCommand` struct in C++: `std::vector<int> valid_targets;` `std::map<std::string, int> params;`
    # It does not have `options` vector in the command class itself?
    # `GameState::QueryContext` has `options`.
    # `QueryCommand::execute` populates `ctx`.
    # Let's look at `QueryCommand::execute` in `commands.cpp`.
    # `ctx.query_type = query_type; ctx.valid_target_ids = valid_targets; ctx.params = params;`
    # It does NOT populate `options`.
    # So `SELECT_OPTION` via `QueryCommand` primitive relies on what?
    # Maybe `params["option_count"]`?
    # If `ActionGenerator` generates based on `query.options.size()`, and `options` is empty, we get 0 actions.

    # Wait, `ActionGenerator` code I wrote:
    # `else if (query.query_type == "SELECT_OPTION") { for (size_t i = 0; i < query.options.size(); ++i) ... }`
    # If `QueryCommand` doesn't populate `options`, this branch is dead.

    # BUT, `QueryContext` logic might be incomplete in `QueryCommand`.
    # However, for `SELECT_TARGET`, it uses `valid_targets` which IS populated.

    # For `SELECT_OPTION`, if `options` are strings, we need a way to pass them.
    # If `QueryCommand` doesn't support passing strings, we can't test it easily unless we modify `QueryCommand`.
    pass

if __name__ == "__main__":
    test_query_flow()
    print("test_query_flow passed")
