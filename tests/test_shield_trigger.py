import json
import os
import sys

# allow imports from project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

import dm_ai_module


def test_shield_trigger_flow():
    # Load JSON card registry
    json_path = os.path.abspath('data/cards.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        js = f.read()
    dm_ai_module.card_registry_load_from_json(js)

    # Load csv card db for other APIs
    card_db = dm_ai_module.CsvLoader.load_cards('data/cards.csv')
    print('Loaded CSV card ids:', list(card_db.keys()))
    if 5 in card_db:
        print('Terror Pit keyword shield_trigger:', card_db[5].keywords.shield_trigger)

    # Setup a game state
    gs = dm_ai_module.GameState(123)
    gs.setup_test_duel()

    # Add an attacker to player 0 battle zone
    attacker_instance = 1001
    gs.add_test_card_to_battle(0, 1, attacker_instance, False, False)  # Card id 1 exists in cards.json

    # Add a defender creature to player 1 battle zone (target for Terror Pit)
    defender_instance = 3001
    gs.add_test_card_to_battle(1, 1, defender_instance, False, False)

    # Add a Terror Pit as a shield on player 1 (id 5)
    # Use set_deck + DevTools.move_cards to reliably place a shield
    gs.set_deck(1, [5])
    dm_ai_module.DevTools.move_cards(gs, 1, dm_ai_module.Zone.DECK, dm_ai_module.Zone.SHIELD, 1, 5)
    print('Before ATTACK:')
    print(' Player0 battle:', [c.instance_id for c in gs.players[0].battle_zone])
    print(' Player1 shield:', [c.instance_id for c in gs.players[1].shield_zone])
    print(' Player1 hand:', [c.instance_id for c in gs.players[1].hand])

    # Attack player: ATTACK_PLAYER then PASS to execute battle
    act = dm_ai_module.Action()
    act.type = dm_ai_module.ActionType.ATTACK_PLAYER
    act.source_instance_id = attacker_instance
    act.target_player = 1
    dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)

    # Now PASS to resolve the battle (this should break the shield and queue shield trigger)
    pass_act = dm_ai_module.Action()
    pass_act.type = dm_ai_module.ActionType.PASS
    dm_ai_module.EffectResolver.resolve_action(gs, pass_act, card_db)

    print('After PASS:')
    print(' Player0 battle:', [c.instance_id for c in gs.players[0].battle_zone])
    print(' Player1 shield:', [c.instance_id for c in gs.players[1].shield_zone])
    print(' Player1 hand:', [c.instance_id for c in gs.players[1].hand])
    print(' Player1 grave:', [c.instance_id for c in gs.players[1].graveyard])
    print(' Pending effects:', dm_ai_module.get_pending_effects_info(gs))

    # At this point, a SHIELD_TRIGGER pending effect should have been pushed and then resolved
    # Resolve the pending SHIELD_TRIGGER (slot 0)
    resolve_act = dm_ai_module.Action()
    resolve_act.type = dm_ai_module.ActionType.RESOLVE_EFFECT
    resolve_act.slot_index = 0
    dm_ai_module.EffectResolver.resolve_action(gs, resolve_act, card_db)
    print('After resolving shield trigger (first resolve):')
    print(' Pending effects:', dm_ai_module.get_pending_effects_info(gs))
    print(' Player1 battle:', [c.instance_id for c in gs.players[1].battle_zone])
    print(' Player1 hand:', [c.instance_id for c in gs.players[1].hand])
    print(' Player1 grave:', [c.instance_id for c in gs.players[1].graveyard])

    # After resolving shield trigger, a new pending effect requiring target selection may be pushed.
    # Select the defender as target for the Terror Pit effect
    sel = dm_ai_module.Action()
    sel.type = dm_ai_module.ActionType.SELECT_TARGET
    sel.slot_index = 0
    sel.target_instance_id = defender_instance
    dm_ai_module.EffectResolver.resolve_action(gs, sel, card_db)
    print('After selecting target:')
    print(' Pending effects:', dm_ai_module.get_pending_effects_info(gs))

    # Finally resolve the destruction pending effect (now at slot 0)
    final_res = dm_ai_module.Action()
    final_res.type = dm_ai_module.ActionType.RESOLVE_EFFECT
    final_res.slot_index = 0
    dm_ai_module.EffectResolver.resolve_action(gs, final_res, card_db)

    print('After final resolve:')
    print(' Pending effects:', dm_ai_module.get_pending_effects_info(gs))
    print(' Player1 battle:', [c.instance_id for c in gs.players[1].battle_zone])
    print(' Player1 grave:', [c.instance_id for c in gs.players[1].graveyard])

    # Check defender moved to graveyard (destroyed)
    opp = gs.players[1]
    in_battle = any(c.instance_id == defender_instance for c in opp.battle_zone)
    in_grave = any(c.instance_id == defender_instance for c in opp.graveyard)

    assert not in_battle, 'Defender should not be in battle zone after Terror Pit'
    assert in_grave, 'Defender should be in graveyard after Terror Pit'


if __name__ == '__main__':
    test_shield_trigger_flow()
    print('Shield trigger flow test completed.')
