
import sys
import os
import pytest
import json

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

@pytest.mark.skipif('dm_ai_module' not in sys.modules, reason="requires dm_ai_module C++ extension")
def test_lethal_puzzle_easy_integration():
    """
    Verification test for 'lethal_puzzle_easy' logic:
    P0 has Speed Attacker (ID 11 in cards.json is SA, ID 12 is also SA).
    The original test used ID 2, but cards.json has specific cards.
    We need to check which card is used in the legacy test logic.
    Legacy test assumed ID 2.
    However, cards.json provided has:
    ID 1: Bronze-Arm Tribe
    ID 7: Oboro
    ID 8: Tigawock
    ID 9: Vibrato
    ID 10: Himecut
    ID 11: Napoleon Vibes (SA, Cost 5, Fire)
    ID 12: Karakuri Barsi (SA, Cost 5, Water/Fire)

    If the legacy test used ID 2, it might have been using a different cards.json or just checking generic logic.
    Since we must use the REAL cards.json, we should pick a card that is actually a Speed Attacker.
    Card ID 11 "Napoleon Vibes" is a Speed Attacker.

    We will set up a scenario where P0 can summon Napoleon Vibes (Cost 5) and win.
    """

    # Load Cards
    if not os.path.exists('data/cards.json'):
        pytest.skip("data/cards.json not found")

    card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

    # Identify a Speed Attacker card ID.
    # From data/cards.json read, ID 11 is "Napoleon Vibes" with "speed_attacker": true. Cost 5.
    sa_card_id = 11

    # Create Game
    gi = dm_ai_module.GameInstance(1, card_db)

    # Configure Scenario
    config = dm_ai_module.ScenarioConfig()
    config.my_mana = 5 # Enough for Cost 5
    config.my_hand_cards = [sa_card_id]
    config.my_mana_zone = [1, 1, 1, 1, 1] # 5 cards (Civilization is Fire? ID 1 is Nature. Need Fire mana)
    # Card 11 is Fire. We need Fire mana.
    # Card 14 (spell side of 11) is Fire.
    # Or we can just assume 5 mana of appropriate color if mana system is strict.
    # ID 1 is Nature.
    # To be safe, we should use ID 11 in mana too if untap, or just rely on 'my_mana' giving available mana count
    # and maybe the engine checks civs.
    # The C++ engine likely checks civs.
    # Let's put ID 11 in mana zone too to ensure Fire mana.
    config.my_mana_zone = [11, 11, 11, 11, 11]

    config.enemy_shield_count = 0 # Lethal puzzle: 0 shields -> direct attack wins

    gi.reset_with_scenario(config)
    state = gi.state

    # Check Setup
    p0 = state.players[0]
    assert len(p0.hand) >= 1
    assert any(c.card_id == sa_card_id for c in p0.hand)

    # 1. Play Speed Attacker
    sa_card = next(c for c in p0.hand if c.card_id == sa_card_id)

    act_play = dm_ai_module.Action()
    act_play.type = dm_ai_module.ActionType.PLAY_CARD
    act_play.card_id = sa_card_id
    act_play.source_instance_id = sa_card.instance_id

    dm_ai_module.EffectResolver.resolve_action(state, act_play, card_db)

    # Resolve Stack/Effects
    # Napoleon Vibes has ON_PLAY trigger (Discard 2, Draw 1).
    # We need to handle this.
    # Loop until stable
    max_steps = 20
    steps = 0
    while steps < max_steps:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
        if not actions:
            break

        # Simple policy: pick first legal action (e.g. resolve effect, discard arbitrary)
        # For Napoleon Vibes: Discard 2 optional.
        # If optional, we might need to choose.
        # If it generates SELECT_TARGET/DISCARD actions, we pick one.
        action = actions[0]
        dm_ai_module.EffectResolver.resolve_action(state, action, card_db)
        steps += 1

    # 2. Attack Player
    # Find creature in Battle Zone
    assert len(p0.battle_zone) > 0
    attacker = p0.battle_zone[-1]
    assert attacker.card_id == sa_card_id
    assert attacker.summoning_sickness == True # Still true, but SA allows attack

    # Generate attack action
    act_attack = dm_ai_module.Action()
    act_attack.type = dm_ai_module.ActionType.ATTACK_PLAYER
    act_attack.source_instance_id = attacker.instance_id
    act_attack.target_player = 1 # Opponent

    # Verify legality (optional but good)
    # legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
    # assert any(a.type == dm_ai_module.ActionType.ATTACK_PLAYER for a in legal_actions)

    dm_ai_module.EffectResolver.resolve_action(state, act_attack, card_db)

    # Resolve Battle (Direct Attack)
    # The engine creates a pending BREAK_SHIELD or similar, or just processes it.
    # Usually requires resolving pending effects.

    steps = 0
    while (state.get_pending_effect_count() > 0) and steps < max_steps:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
        if not actions:
            break
        action = actions[0]
        dm_ai_module.EffectResolver.resolve_action(state, action, card_db)
        steps += 1

    # Check Winner
    assert state.winner == dm_ai_module.GameResult.P1_WIN # P1 is player index 0 (GameResult 1)
