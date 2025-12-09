
import sys
import os
import pytest

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

try:
    import dm_ai_module
    from dm_ai_module import GameState, CardDefinition, CardType, Civilization, GameResult, ActionType, ActionGenerator, CardData
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def create_mock_card_db():
    card_db = {}
    # Card 1: Bronze Arm Tribe (Creature, Cost 3)
    c1 = CardDefinition()
    c1.id = 1
    c1.name = "Bronze Arm Tribe"
    c1.type = CardType.CREATURE
    c1.civilization = Civilization.NATURE
    c1.cost = 3
    c1.power = 1000
    card_db[1] = c1

    # Card 2: Aqua Hulcus (Creature, Cost 3)
    c2 = CardDefinition()
    c2.id = 2
    c2.name = "Aqua Hulcus"
    c2.type = CardType.CREATURE
    c2.civilization = Civilization.WATER
    c2.cost = 3
    c2.power = 2000
    card_db[2] = c2
    return card_db

def test_card_stats_tracking():
    """
    Verifies that playing a card correctly updates its usage statistics in GameState.
    """
    gs = GameState(42)
    card_db = create_mock_card_db()

    # Register cards
    for cid, cdef in card_db.items():
        cdata = CardData(cid, cdef.name, cdef.cost, cdef.civilization.name, cdef.power,
                         "CREATURE" if cdef.type == CardType.CREATURE else "SPELL", [], [])
        dm_ai_module.register_card_data(cdata)

    # Initialize stats for all cards in DB
    dm_ai_module.initialize_card_stats(gs, card_db, 40)

    gi = dm_ai_module.GameInstance(42, card_db)
    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [1] # Card 1 in hand
    config.my_mana_zone = [1] * 3 # 3 Mana
    gi.reset_with_scenario(config)

    state = gi.state
    state.current_phase = dm_ai_module.Phase.MAIN

    # Initialize stats on this new state
    dm_ai_module.initialize_card_stats(state, card_db, 40)

    # Generate play action
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    play_action = None
    for action in actions:
        if (action.type == ActionType.PLAY_CARD or action.type == ActionType.DECLARE_PLAY) and action.card_id == 1:
            play_action = action
            break

    assert play_action is not None

    # Execute action
    dm_ai_module.EffectResolver.resolve_action(state, play_action, card_db)
    
    if play_action.type == ActionType.DECLARE_PLAY:
        # Step 2: PAY_COST
        actions = ActionGenerator.generate_legal_actions(state, card_db)
        pay_action = next((a for a in actions if a.type == ActionType.PAY_COST), None)
        if pay_action:
             dm_ai_module.EffectResolver.resolve_action(state, pay_action, card_db)
             
        # Step 3: RESOLVE_PLAY
        actions = ActionGenerator.generate_legal_actions(state, card_db)
        resolve_action = next((a for a in actions if a.type == ActionType.RESOLVE_PLAY), None)
        if resolve_action:
             dm_ai_module.EffectResolver.resolve_action(state, resolve_action, card_db)


    # Check stats again
    stats_after = dm_ai_module.get_card_stats(state)
    assert 1 in stats_after
    # Use attribute access, not dict access
    assert stats_after[1].play_count == 1
    assert stats_after[1].sum_cost_discount == 0 # Cost 3 played with 3 mana

    # Check early usage
    # Scenario defaults to Turn 5.
    assert state.turn_number == 5
    # So early_usage should be 0.
    assert stats_after[1].sum_early_usage == 0.0

def test_card_stats_win_contribution():
    """
    Verifies that winning a game updates win contribution stats.
    """
    card_db = create_mock_card_db()

    # Register cards
    for cid, cdef in card_db.items():
        cdata = CardData(cid, cdef.name, cdef.cost, cdef.civilization.name, cdef.power,
                         "CREATURE" if cdef.type == CardType.CREATURE else "SPELL", [], [])
        dm_ai_module.register_card_data(cdata)

    gi = dm_ai_module.GameInstance(42, card_db)

    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [2] # Aqua Hulcus
    config.my_mana_zone = [2] * 3
    config.my_battle_zone = [1] # Bronze Arm Tribe (attacker)
    config.enemy_shield_count = 0
    gi.reset_with_scenario(config)

    state = gi.state
    # Initialize stats
    dm_ai_module.initialize_card_stats(state, card_db, 40)

    # Play Aqua Hulcus (Card 2)
    state.current_phase = dm_ai_module.Phase.MAIN
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    play_action = [a for a in actions if (a.type == ActionType.PLAY_CARD or a.type == ActionType.DECLARE_PLAY) and a.card_id == 2][0]
    dm_ai_module.EffectResolver.resolve_action(state, play_action, card_db)

    if play_action.type == ActionType.DECLARE_PLAY:
        # Pay Cost
        actions = ActionGenerator.generate_legal_actions(state, card_db)
        pay_action = [a for a in actions if a.type == ActionType.PAY_COST][0]
        dm_ai_module.EffectResolver.resolve_action(state, pay_action, card_db)
        # Resolve Play
        actions = ActionGenerator.generate_legal_actions(state, card_db)
        resolve_action = [a for a in actions if a.type == ActionType.RESOLVE_PLAY][0]
        dm_ai_module.EffectResolver.resolve_action(state, resolve_action, card_db)


    # Proceed to Attack Phase
    state.current_phase = dm_ai_module.Phase.ATTACK

    # Attack Player with Card 1
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    attack_action = None
    for a in actions:
        if a.type == ActionType.ATTACK_PLAYER:
            attack_action = a
            break

    assert attack_action is not None

    dm_ai_module.EffectResolver.resolve_action(state, attack_action, card_db)

    # Check pending effects (BREAK_SHIELD or similar)
    # Since enemy has 0 shields, it might be direct attack logic handled in execute_battle or similar.
    # But usually it queues BREAK_SHIELD which handles game over if no shields.

    # Or maybe RESOLVE_BATTLE.

    pending = dm_ai_module.get_pending_effects_info(state)

    # Resolve pending effects loop
    limit = 10
    while limit > 0 and state.winner == dm_ai_module.GameResult.NONE:
         actions = ActionGenerator.generate_legal_actions(state, card_db)
         if not actions: break

         # Prioritize resolution actions
         res_act = next((a for a in actions if a.type in [ActionType.RESOLVE_BATTLE, ActionType.BREAK_SHIELD, ActionType.RESOLVE_EFFECT]), None)
         if res_act:
             dm_ai_module.EffectResolver.resolve_action(state, res_act, card_db)
         else:
             # Just take any action (PASS?)
             dm_ai_module.EffectResolver.resolve_action(state, actions[0], card_db)
         limit -= 1

    assert state.winner == dm_ai_module.GameResult.P1_WIN

    # Check stats for Card 2
    stats = dm_ai_module.get_card_stats(state)
    assert 2 in stats
    # Use attribute access
    assert stats[2].play_count == 1
    assert stats[2].win_count == 1
    assert stats[2].sum_win_contribution == 1.0
