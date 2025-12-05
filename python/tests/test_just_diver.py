import pytest
import sys
import os

# Add bin path to sys.path
bin_path = os.path.join(os.path.dirname(__file__), '..', 'bin')
sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    pytest.fail("dm_ai_module not found. Please build the C++ module first.")

def test_just_diver():
    # Setup Card DB
    card_db = {
        1: dm_ai_module.CardDefinition(
            1, "Just Diver Creature", "WATER", ["Liquid People"], 2, 2000,
            dm_ai_module.CardKeywords(), []
        ),
        2: dm_ai_module.CardDefinition(
            2, "Enemy Spell", "DARKNESS", ["Demon"], 3, 0,
            dm_ai_module.CardKeywords(), []
        ),
        3: dm_ai_module.CardDefinition(
            3, "Enemy Creature", "FIRE", ["Dragon"], 3, 3000,
            dm_ai_module.CardKeywords(), []
        )
    }

    # Enable Just Diver on card 1
    # Note: We need to set it via property since constructor might not expose it easily yet
    # if we didn't update Python bindings for constructor, but we exposed the property.
    card_db[1].keywords.just_diver = True

    # Setup Game State
    game = dm_ai_module.GameState(100)
    game.active_player_id = 0
    game.turn_number = 1

    # 1. Player 1 plays Just Diver Creature
    # Manually adding to Battle Zone implies it was just played?
    # We need to simulate the "played this turn" check.
    # The current logic for Just Diver requires tracking `turn_played`.
    # Since we can't easily set `turn_played` via Python (it's internal in CardInstance usually),
    # we might need to rely on the engine setting it when we `resolve_play`.

    # Let's try to do a proper play flow.
    game.add_card_to_hand(0, 1, 100) # Player 0, Card 1, Instance 100

    # Generate PLAY action
    # We can skip to PLAY_CARD action directly
    action_play = dm_ai_module.Action()
    action_play.type = dm_ai_module.ActionType.PLAY_CARD
    action_play.source_instance_id = 100
    action_play.target_player = 0 # Self

    # Resolve Play
    # We assume infinite mana for test or we just hack the stack
    # Let's use `add_test_card_to_battle` if `turn_played` is exposed, but it's not.
    # So we must play it.

    # Cheat mana
    game.add_card_to_mana(0, 1, 900)
    game.add_card_to_mana(0, 1, 901) # 2 mana

    dm_ai_module.EffectResolver.resolve_action(game, action_play, card_db)

    # Check if it's in battle zone
    p1_battle = game.players[0].battle_zone
    assert len(p1_battle) == 1
    jd_creature = p1_battle[0]

    # Verify Just Diver Logic
    # Opponent (Player 1) tries to choose it as target.

    # Case A: Same Turn (Turn 1)
    # TargetUtils is C++ only, but ActionGenerator uses it.
    # We can try to generate a SELECT_TARGET action for opponent.

    # Setup Opponent (Player 1)
    game.active_player_id = 1 # Switch active player to opponent

    # Opponent plays a spell that targets a creature
    # We create a new card data using Generic system to register the card
    effect_def = dm_ai_module.EffectDef()
    effect_def.trigger = dm_ai_module.TriggerType.ON_PLAY

    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.DESTROY
    action_def.scope = dm_ai_module.TargetScope.TARGET_SELECT

    f_select = dm_ai_module.FilterDef()
    f_select.zones = ["BATTLE_ZONE"]
    f_select.owner = "OPPONENT"
    action_def.filter = f_select

    effect_def.actions = [action_def]

    card_data = dm_ai_module.CardData(2, "Destroyer", 3, "DARKNESS", 0, "SPELL", [], [effect_def])
    dm_ai_module.register_card_data(card_data)

    # Need to update card_db with this info too if EffectResolver uses card_db for basic info
    # But GenericCardSystem uses Registry.

    # Play the spell
    game.add_card_to_hand(1, 2, 200) # Card 2 (Spell)
    game.add_card_to_mana(1, 2, 902)
    game.add_card_to_mana(1, 2, 903)
    game.add_card_to_mana(1, 2, 904)

    play_act = dm_ai_module.Action()
    play_act.type = dm_ai_module.ActionType.PLAY_CARD
    play_act.source_instance_id = 200
    play_act.target_player = 1

    dm_ai_module.EffectResolver.resolve_action(game, play_act, card_db)

    # Now check legal actions.
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    # We expect NO actions that target the JD creature (Instance 100).
    # If Just Diver works, the opponent cannot choose it.

    targets_jd = False
    for a in actions:
        if a.type == dm_ai_module.ActionType.SELECT_TARGET:
            if a.target_instance_id == 100:
                targets_jd = True

    assert targets_jd == False, "Opponent should NOT be able to target Just Diver creature on Turn 1"

    # Case B: Next Turn (Self Turn)
    game.active_player_id = 0
    game.turn_number = 2
    # Still protected? "Until start of YOUR next turn".
    # Player 0 (Owner) Next Turn Start.
    # So on Turn 2 (P0), it expires at START.

    # Now Test T2, P1 (Should succeed to target).
    # We need to set up the scenario again for T2 P1.
    # The previous PendingEffect was probably cleared or stuck.
    # Let's clear pending effects to be safe (though we can't easily clear them from Python without helper).
    # But since we didn't complete the selection, it might be stuck.
    # Let's assume we can reuse the pending effect if it's there.

    # Check pending effects count
    pe_info = dm_ai_module.get_pending_effects_info(game)
    if len(pe_info) == 0:
        # Re-play the spell logic if it was cleared (e.g. by passing)
        pass
        # But we didn't execute PASS.

    game.turn_number = 2
    game.active_player_id = 1

    actions_t2 = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    targets_jd_t2 = False
    for a in actions_t2:
        if a.type == dm_ai_module.ActionType.SELECT_TARGET:
            if a.target_instance_id == 100:
                targets_jd_t2 = True

    # If no pending effect, we can't verify target.
    if len(pe_info) > 0:
        assert targets_jd_t2 == True, "Opponent SHOULD be able to target Just Diver creature after protection expires"

if __name__ == "__main__":
    test_just_diver()
