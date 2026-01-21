import pytest
import dm_ai_module
from dm_ai_module import GameInstance, Action, ActionType, CardType, CardStub

def test_spell_cast_and_stack_resolution():
    """
    Verifies that casting a spell:
    1. Removes it from hand.
    2. Puts it on the pending_effects stack (not graveyard yet).
    3. Resolving the effect moves it to graveyard.
    """
    game = GameInstance()
    game.start_game()

    # Setup: Player 1 (index 0) is active
    p1 = game.state.players[0]

    # Clear hand and add a Spell card (ID 2 is "Test Spell" in stub)
    p1.hand = []
    spell_instance_id = 100
    p1.hand.append(CardStub(2, spell_instance_id))

    # Verify initial state
    assert len(p1.hand) == 1
    assert len(p1.graveyard) == 0
    assert len(game.state.pending_effects) == 0

    # Action: Cast Spell (PLAY_CARD)
    action = Action()
    action.type = ActionType.PLAY_CARD
    action.source_instance_id = spell_instance_id

    game.execute_action(action)

    # Verify after cast
    # 1. Removed from hand
    assert len(p1.hand) == 0
    # 2. Not in graveyard yet
    assert len(p1.graveyard) == 0
    # 3. On stack
    assert len(game.state.pending_effects) == 1
    effect = game.state.pending_effects[0]
    assert effect["type"] == "SPELL_EFFECT"
    assert effect["source_id"] == spell_instance_id

    # Action: Resolve Effect
    resolve_action = Action()
    resolve_action.type = ActionType.RESOLVE_EFFECT

    game.execute_action(resolve_action)

    # Verify after resolution
    # 1. Stack empty
    assert len(game.state.pending_effects) == 0
    # 2. In graveyard
    assert len(p1.graveyard) == 1
    assert p1.graveyard[0].instance_id == spell_instance_id

def test_creature_summon_no_stack():
    """
    Verifies that summoning a creature:
    1. Removes it from hand.
    2. Puts it in Battle Zone immediately (no stack for summon itself in this simple stub).
    """
    game = GameInstance()
    game.start_game()

    p1 = game.state.players[0]
    p1.hand = []
    creature_instance_id = 200
    p1.hand.append(CardStub(1, creature_instance_id)) # ID 1 is "Test Creature"

    # Action: Play Creature
    action = Action()
    action.type = ActionType.PLAY_CARD
    action.source_instance_id = creature_instance_id

    game.execute_action(action)

    # Verify
    assert len(p1.hand) == 0
    assert len(game.state.pending_effects) == 0
    assert len(p1.battle_zone) == 1
    assert p1.battle_zone[0].instance_id == creature_instance_id

def test_multiple_stack_resolution():
    """
    Verifies LIFO processing of stack.
    """
    game = GameInstance()
    game.state.pending_effects = []

    # Manually push effects
    game.state.pending_effects.append({"type": "EFFECT_1", "id": 1})
    game.state.pending_effects.append({"type": "EFFECT_2", "id": 2})

    # Resolve first (should be EFFECT_2)
    resolve_action = Action()
    resolve_action.type = ActionType.RESOLVE_EFFECT
    game.execute_action(resolve_action)

    assert len(game.state.pending_effects) == 1
    assert game.state.pending_effects[0]["id"] == 1

    # Resolve second (should be EFFECT_1)
    game.execute_action(resolve_action)
    assert len(game.state.pending_effects) == 0
