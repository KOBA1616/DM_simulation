import pytest
from dm_ai_module import GameState, Player, CardInstance, CardDefinition, CardType, Civilization, Zone, Phase, Action, ActionType, EffectType, GameInstance, JsonLoader, CardKeywords, EffectResolver, FilterDef

def test_revolution_change_condition():
    """
    Test Revolution Change with FilterDef conditions.
    """

    # 1. Define Attacker (Water Bird)
    water_attacker_def = CardDefinition()
    water_attacker_def.name = "Water Attacker"
    water_attacker_def.id = 101
    water_attacker_def.civilization = Civilization.WATER
    water_attacker_def.races = ["Cyber Virus"]
    water_attacker_def.power = 1000
    water_attacker_def.cost = 2
    water_attacker_def.type = CardType.CREATURE

    # 2. Define Attacker (Fire Bird)
    fire_attacker_def = CardDefinition()
    fire_attacker_def.name = "Fire Attacker"
    fire_attacker_def.id = 102
    fire_attacker_def.civilization = Civilization.FIRE
    fire_attacker_def.races = ["Fire Bird"]
    fire_attacker_def.power = 1000
    fire_attacker_def.cost = 2
    fire_attacker_def.type = CardType.CREATURE

    # 3. Define Rev Change User (Requires FIRE)
    rev_change_def = CardDefinition()
    rev_change_def.name = "RevChange Dragon"
    rev_change_def.id = 200
    rev_change_def.civilization = Civilization.FIRE
    rev_change_def.races = ["Mega Command Dragon"]
    rev_change_def.power = 7000
    rev_change_def.cost = 7
    rev_change_def.type = CardType.CREATURE

    k = CardKeywords()
    k.revolution_change = True
    rev_change_def.keywords = k

    # Set Condition: Must be FIRE
    f = FilterDef()
    f.civilizations = ["FIRE"]
    rev_change_def.revolution_change_condition = f

    card_db = {
        101: water_attacker_def,
        102: fire_attacker_def,
        200: rev_change_def
    }

    # --- TEST CASE 1: Invalid Attacker (Water) ---
    game1 = GameInstance(0, card_db)
    state1 = game1.state
    state1.active_player_id = 0
    state1.current_phase = Phase.MAIN

    state1.add_test_card_to_battle(0, 101, 10, False, False) # Water Attacker
    state1.add_card_to_hand(0, 200, 20) # Rev Change Card

    # Attack with Water Bird
    attack_action = Action()
    attack_action.type = ActionType.ATTACK_PLAYER
    attack_action.source_instance_id = 10
    attack_action.target_player = 1

    EffectResolver.resolve_action(state1, attack_action, card_db)

    # Should NOT trigger pending effect
    from dm_ai_module import get_pending_effects_info
    pending1 = get_pending_effects_info(state1)
    assert len(pending1) == 0, "Water attacker should NOT trigger Fire Revolution Change"
    assert state1.current_phase == Phase.BLOCK, "Should go directly to block phase"

    print("Test Case 1 Passed: Invalid Condition check.")

    # --- TEST CASE 2: Valid Attacker (Fire) ---
    game2 = GameInstance(0, card_db)
    state2 = game2.state
    state2.active_player_id = 0
    state2.current_phase = Phase.MAIN

    state2.add_test_card_to_battle(0, 102, 10, False, False) # Fire Attacker
    state2.add_card_to_hand(0, 200, 20) # Rev Change Card

    # Attack with Fire Bird
    attack_action2 = Action()
    attack_action2.type = ActionType.ATTACK_PLAYER
    attack_action2.source_instance_id = 10
    attack_action2.target_player = 1

    EffectResolver.resolve_action(state2, attack_action2, card_db)

    # Should TRIGGER pending effect
    pending2 = get_pending_effects_info(state2)
    assert len(pending2) > 0, "Fire attacker SHOULD trigger Fire Revolution Change"
    assert pending2[0][0] == int(EffectType.ON_ATTACK_FROM_HAND)

    print("Test Case 2 Passed: Valid Condition check.")

if __name__ == "__main__":
    test_revolution_change_condition()
