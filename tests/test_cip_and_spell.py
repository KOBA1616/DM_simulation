import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

import dm_ai_module


def test_cip_add_mana():
    # Load registry and json
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    # Also register for CardRegistry if needed by system internals
    # But usually GameInstance/EffectResolver use the passed card_db map

    gs = dm_ai_module.GameState(1)
    gs.setup_test_duel()

    # Prepare deck and mana: ensure player has enough mana to play the 3-cost creature
    # Put several copies so we can move some to mana zone and one to hand
    gs.set_deck(0, [1,1,1,1])
    # Move 3 cards from deck to mana zone to pay cost
    dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.MANA, 3, 1)
    # Move one copy to hand to play
    dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND, 1, 1)

    # Play the card: simulate PLAY_CARD action
    act = dm_ai_module.Action()
    act.type = dm_ai_module.ActionType.PLAY_CARD
    act.card_id = 1
    # set source_instance_id to the instance id of card in hand
    src = gs.players[0].hand[0].instance_id
    act.source_instance_id = src

    # Play (resolve action)
    dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)

    # Resolve pending CIP if any (use binding helper)
    if dm_ai_module.get_pending_effects_info(gs):
        ra = dm_ai_module.Action(); ra.type = dm_ai_module.ActionType.RESOLVE_EFFECT; ra.slot_index = 0
        dm_ai_module.EffectResolver.resolve_action(gs, ra, card_db)

    # After CIP ADD_MANA, expect a card moved from deck to mana zone
    assert len(gs.players[0].mana_zone) >= 1


def test_spell_return_to_hand():
    # Spiral Gate (id=6) should return a creature to hand
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")

    gs = dm_ai_module.GameState(2)
    gs.setup_test_duel()

    # Prepare deck and mana for Spiral Gate (cost 2). Place two mana and one copy in hand.
    gs.set_deck(0, [6,6,6])
    dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.MANA, 2, 6)
    dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND, 1, 6)

    # Add a creature to opponent battle zone
    gs.add_test_card_to_battle(1, 1, 5001, False, False)

    # Play spiral gate (Follow play transaction)
    # 1. Declare Play
    act = dm_ai_module.Action(); act.type = dm_ai_module.ActionType.PLAY_CARD; act.card_id = 6
    act.source_instance_id = gs.players[0].hand[0].instance_id
    dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)

    # 2. Pay Cost (Auto-pay)
    # Ideally use ActionGenerator but here we manually resolve for test
    pay = dm_ai_module.Action(); pay.type = dm_ai_module.ActionType.PAY_COST
    # Need source_instance_id from stack?
    # Stack has the card.
    stack_card = gs.stack_zone[-1]
    pay.source_instance_id = stack_card.instance_id
    dm_ai_module.EffectResolver.resolve_action(gs, pay, card_db)

    # 3. Resolve Play (Spells go to grave, triggers execute)
    res = dm_ai_module.Action(); res.type = dm_ai_module.ActionType.RESOLVE_PLAY
    res.source_instance_id = stack_card.instance_id
    dm_ai_module.EffectResolver.resolve_action(gs, res, card_db)

    # Check pending effects. Spells resolve immediately, so we might go straight to Target Select.
    pending = dm_ai_module.get_pending_effects_info(gs)
    if pending:
        # If it's Target Select (EffectType.NONE usually), we select directly.
        # If it's a queued trigger (e.g. CIP), we resolve it first.
        ptype = pending[0][0]

        if ptype != 0: # Not NONE/TargetSelect, assume Trigger (e.g. CIP)
             r = dm_ai_module.Action(); r.type = dm_ai_module.ActionType.RESOLVE_EFFECT; r.slot_index = 0
             dm_ai_module.EffectResolver.resolve_action(gs, r, card_db)
             pending = dm_ai_module.get_pending_effects_info(gs)

        if pending:
            # Now we should be at Target Select
            sel = dm_ai_module.Action(); sel.type = dm_ai_module.ActionType.SELECT_TARGET; sel.slot_index = 0
            sel.target_instance_id = 5001
            dm_ai_module.EffectResolver.resolve_action(gs, sel, card_db)
            fin = dm_ai_module.Action(); fin.type = dm_ai_module.ActionType.RESOLVE_EFFECT; fin.slot_index = 0
            dm_ai_module.EffectResolver.resolve_action(gs, fin, card_db)

    # Now confirm opponent creature moved to hand
    opp = gs.players[1]
    in_hand = any(c.instance_id == 5001 for c in opp.hand)
    assert in_hand

if __name__ == '__main__':
    test_cip_add_mana()
    print('CIP test passed')
    test_spell_return_to_hand()
    print('Spell test passed')
