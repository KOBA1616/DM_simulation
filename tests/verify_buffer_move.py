
import sys
import os

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
sys.path.append('bin')

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure it is built.")
    sys.exit(1)

def test_buffer_move():
    print("Testing MOVE_CARD from effect_buffer...")

    # Setup state
    state = dm_ai_module.GameState(100)

    player_id = 0
    card_id = 1
    instance_id = 999

    # Add to buffer
    state.add_test_card_to_buffer(player_id, card_id, instance_id)

    # Verify add
    assert len(state.players[player_id].effect_buffer) == 1

    # Let's use `resolve_effect_with_targets`.
    action_def = dm_ai_module.ActionDef()
    # Correct enum type
    action_def.type = dm_ai_module.EffectActionType.MOVE_CARD
    action_def.destination_zone = "GRAVEYARD"

    effect_def = dm_ai_module.EffectDef()
    effect_def.actions = [action_def]

    # We need a db.
    card_db = {}

    # Context
    ctx = {}

    # Invoke
    dm_ai_module.GenericCardSystem.resolve_effect_with_targets(
        state,
        effect_def,
        [instance_id],
        0, # source_id
        card_db,
        ctx
    )

    # Verify
    # Buffer should be empty
    assert len(state.players[player_id].effect_buffer) == 0, "Buffer not empty"

    # Graveyard should have the card
    graveyard = state.players[player_id].graveyard
    assert len(graveyard) == 1, "Graveyard empty"
    assert graveyard[0].instance_id == instance_id, "Wrong card in graveyard"

    print("  Buffer move passed.")

if __name__ == "__main__":
    test_buffer_move()
    print("All tests passed.")
