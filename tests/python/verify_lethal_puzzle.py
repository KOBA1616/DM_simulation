
import os
import sys
sys.path.append(os.path.abspath('bin'))
import dm_ai_module
from dm_toolkit.training.scenario_definitions import get_scenario_config

def test_lethal_puzzle_logic():
    print("Testing lethal_puzzle_easy logic...")

    # Load cards
    with open('data/cards.json', 'r', encoding='utf-8') as f:
        json_content = f.read()
        dm_ai_module.card_registry_load_from_json(json_content)

    # Use JsonLoader to get the card_db map consistent with cards.json
    card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

    # Load scenario config
    config = get_scenario_config('lethal_puzzle_easy')

    gi = dm_ai_module.GameInstance(42, card_db)
    gi.reset_with_scenario(config)

    state = gi.state
    p0 = state.players[0]
    p1 = state.players[1]

    print(f"P0 Hand: {[c.card_id for c in p0.hand]}")
    print(f"P0 Battle: {[c.card_id for c in p0.battle_zone]}")
    print(f"P0 Mana: {[c.card_id for c in p0.mana_zone]}")
    print(f"P0 Mana Tapped: {[c.is_tapped for c in p0.mana_zone]}")
    print(f"P1 Shields: {len(p1.shield_zone)}")

    # Verify setup: Hand has ID 2 (Speed Attacker), Mana has 3 cards.
    if 2 not in [c.card_id for c in p0.hand]:
        print("FAIL: Speed Attacker (ID 2) not in hand.")
        sys.exit(1)

    # 1. Summon Speed Attacker
    sa_card = next(c for c in p0.hand if c.card_id == 2)
    print("Summoning Speed Attacker...")
    act_play = dm_ai_module.Action()
    act_play.type = dm_ai_module.ActionType.PLAY_CARD
    act_play.card_id = 2
    act_play.source_instance_id = sa_card.instance_id
    dm_ai_module.EffectResolver.resolve_action(state, act_play, card_db)

    # Handle Stack (PAY_COST -> RESOLVE_PLAY)
    while state.stack_zone:
        # Generate legal actions (should be PAY_COST or RESOLVE_PLAY)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
        if not actions:
            print("FAIL: Stack not empty but no actions.")
            sys.exit(1)
        # Pick first action (usually only one for stack)
        action = actions[0]
        dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

    # Handle Pending Effects (e.g. ON_PLAY)
    while state.pending_effects:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db)
        if not actions:
            break
        action = actions[0]
        dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

    # 2. Attack with Speed Attacker
    # Find new instance in BZ
    if not p0.battle_zone:
        print("FAIL: No cards in battle zone after summon.")
        sys.exit(1)

    sa_instance = p0.battle_zone[-1] # last one added
    if sa_instance.card_id != 2:
        print("FAIL: Speed Attacker not found in BZ")
        sys.exit(1)

    if sa_instance.summoning_sickness:
        print("FAIL: Speed Attacker has summoning sickness!")
        # This checks if the fix works
        sys.exit(1)

    print("Attacking with Speed Attacker...")
    act_att2 = dm_ai_module.Action()
    act_att2.type = dm_ai_module.ActionType.ATTACK_PLAYER
    act_att2.source_instance_id = sa_instance.instance_id
    act_att2.target_player = 1
    dm_ai_module.EffectResolver.resolve_action(state, act_att2, card_db)

    # Pass to resolve (Break Shield / Direct Attack)
    pass_act = dm_ai_module.Action()
    pass_act.type = dm_ai_module.ActionType.PASS
    dm_ai_module.EffectResolver.resolve_action(state, pass_act, card_db)

    if state.winner == dm_ai_module.GameResult.P1_WIN:
        print("SUCCESS: Player 1 Won!")
    else:
        print(f"FAIL: Game Result {state.winner}")
        print(f"P1 Battle: {[c.card_id for c in p0.battle_zone]}")
        sys.exit(1)

if __name__ == "__main__":
    test_lethal_puzzle_logic()
