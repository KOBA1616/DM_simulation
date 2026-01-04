
import pytest
import dm_ai_module

def test_restriction_system_blocks_attack():
    gs = dm_ai_module.GameState(100)
    gs.players[0].battle_zone = []
    gs.players[1].battle_zone = []

    # Add a passive effect that forbids attacks
    pe = dm_ai_module.PassiveEffect()
    pe.type = dm_ai_module.PassiveType.CANNOT_ATTACK
    pe.controller = 0

    # Use helper method to avoid copy semantics
    gs.add_passive_effect(pe)

    # Setup attacker
    card_data = dm_ai_module.CardData(1, "Attacker", 1, dm_ai_module.Civilization.FIRE, 1000, dm_ai_module.CardType.CREATURE, [], [])
    dm_ai_module.register_card_data(card_data)

    card_db = dm_ai_module.CardRegistry.get_all_cards()

    attacker = dm_ai_module.CardInstance()
    attacker.instance_id = 0
    attacker.card_id = 1
    attacker.owner = 0
    attacker.is_tapped = False
    attacker.summoning_sickness = False

    gs.players[0].battle_zone.append(attacker)
    gs.register_card_instance(attacker)

    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.ATTACK_PLAYER
    action.source_instance_id = 0
    action.target_instance_id = -1

    history_len_before = len(gs.command_history)

    dm_ai_module.EffectResolver.resolve_action(gs, action, card_db)

    history_len_after = len(gs.command_history)

    print(f"Attack Test: Before={history_len_before}, After={history_len_after}")

    assert history_len_after == history_len_before, f"Attack should be blocked. Diff={history_len_after - history_len_before}"

def test_restriction_system_blocks_play():
    gs = dm_ai_module.GameState(100)

    # Register card
    card_data = dm_ai_module.CardData(2, "Forbidden", 1, dm_ai_module.Civilization.FIRE, 1000, dm_ai_module.CardType.CREATURE, [], [])
    dm_ai_module.register_card_data(card_data)
    card_db = dm_ai_module.CardRegistry.get_all_cards()

    # Add to hand
    hand_card = dm_ai_module.CardInstance()
    hand_card.instance_id = 1
    hand_card.card_id = 2
    hand_card.owner = 0

    gs.players[0].hand.append(hand_card)
    gs.register_card_instance(hand_card)

    # Add passive effect CANNOT_SUMMON
    pe = dm_ai_module.PassiveEffect()
    pe.type = dm_ai_module.PassiveType.CANNOT_SUMMON
    pe.controller = 0

    # Use helper method
    gs.add_passive_effect(pe)

    # Try to play
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.PLAY_CARD
    action.source_instance_id = hand_card.instance_id

    history_len_before = len(gs.command_history)

    dm_ai_module.EffectResolver.resolve_action(gs, action, card_db)

    history_len_after = len(gs.command_history)

    print(f"Play Test: Before={history_len_before}, After={history_len_after}")

    assert history_len_after == history_len_before, f"Play should be blocked. Diff={history_len_after - history_len_before}"
