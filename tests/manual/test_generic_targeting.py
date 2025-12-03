
import pytest
import sys
import os

# Add bin to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def test_generic_targeting_tap():
    """
    Test a custom card that uses Generic Targeting to TAP an opponent's creature.
    """
    # 1. Define a custom card with generic targeting
    # Card ID 9000: "Tap Spell"
    # Effect: Select 1 Opponent Creature in BattleZone -> Tap it.

    tap_effect = dm_ai_module.EffectDef(
        trigger=dm_ai_module.TriggerType.ON_PLAY, # Spells usually use ON_PLAY or main execution
        condition=dm_ai_module.ConditionDef(),
        actions=[
            dm_ai_module.ActionDef(
                type=dm_ai_module.EffectActionType.TAP,
                scope=dm_ai_module.TargetScope.TARGET_SELECT,
                filter=dm_ai_module.FilterDef(
                    owner="OPPONENT",
                    zones=["BATTLE_ZONE"],
                    types=["CREATURE"],
                    count=1
                )
            )
        ]
    )

    # Register to CardRegistry (Required for GenericCardSystem)
    card_data = dm_ai_module.CardData(
        id=9000,
        name="Tap Spell",
        cost=1,
        civilization="LIGHT",
        power=0,
        type="SPELL",
        races=["Spell"],
        effects=[tap_effect]
    )
    dm_ai_module.register_card_data(card_data)

    # Also need CardDefinition for card_db (used by ActionGenerator/GameState validation)
    cd_9000 = dm_ai_module.CardDefinition(
        id=9000,
        name="Tap Spell",
        civilization="LIGHT",
        races=["Spell"],
        cost=1,
        power=0,
        keywords=dm_ai_module.CardKeywords(),
        effects=[tap_effect]
    )
    # Important: Set type to SPELL because binding constructor doesn't take it and defaults to CREATURE (0)
    # EffectResolver only triggers ON_PLAY for Spells or Creatures with CIP keyword.
    cd_9000.type = dm_ai_module.CardType.SPELL

    cd_9001 = dm_ai_module.CardDefinition(
        id=9001,
        name="Dummy Creature",
        civilization="FIRE",
        races=["Human"],
        cost=2,
        power=2000,
        keywords=dm_ai_module.CardKeywords(),
        effects=[]
    )
    # Explicitly set type to CREATURE (though default is 0/CREATURE, being explicit is safer)
    cd_9001.type = dm_ai_module.CardType.CREATURE

    card_db = {
        9000: cd_9000,
        9001: cd_9001
    }

    # 2. Setup Game
    game = dm_ai_module.GameInstance()
    game.start_game(card_db)   # Exists in binding

    # Force Main Phase for testing
    game.state.current_phase = dm_ai_module.Phase.MAIN
    game.state.active_player_id = 0

    # Player 0 has the Spell
    game.state.add_card_to_hand(0, 9000, 100)
    game.state.add_card_to_mana(0, 9000, 101) # Mana source

    # Player 1 has a Creature (Untapped)
    game.state.add_card_to_battle(1, 9001, 200)
    # Note: Accessing battle_zone returns a copy, so modifying it here has no effect on C++ state.
    # But add_card_to_battle initializes it correctly.

    # Check initial state
    print(f"Initial Tapped State: {game.state.players[1].battle_zone[0].is_tapped}")

    # 3. Player 0 Plays the Spell
    # Main Phase -> Generate Actions -> Play Card 9000
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game.state, card_db)
    play_action = None
    for a in actions:
        if a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 9000:
            play_action = a
            break

    assert play_action is not None, "Should be able to play the spell"

    # Execute Play
    dm_ai_module.EffectResolver.resolve_action(game.state, play_action, card_db)

    # 4. Expect Pending Effect (Target Selection)
    # The engine should pause and ask for target selection.

    # Check pending effects
    pe_list = dm_ai_module.get_pending_effects_verbose(game.state)
    assert len(pe_list) > 0, "Should have pending effect"
    # tuple: (type, source, controller, needed, selected, has_def)
    assert pe_list[0][3] == 1, "Should need 1 target"

    actions_select = dm_ai_module.ActionGenerator.generate_legal_actions(game.state, card_db)

    # Check if we have SELECT_TARGET actions
    select_actions = [a for a in actions_select if a.type == dm_ai_module.ActionType.SELECT_TARGET]
    assert len(select_actions) > 0, "Should have selection actions"

    # The target should be the opponent's creature (ID 200)
    target_action = None
    for a in select_actions:
        if a.target_instance_id == 200:
            target_action = a
            break

    assert target_action is not None, "Should be able to select opponent creature"

    # 5. Execute Selection
    dm_ai_module.EffectResolver.resolve_action(game.state, target_action, card_db)

    # 6. Verify Selection State
    pe_list_after = dm_ai_module.get_pending_effects_verbose(game.state)
    assert len(pe_list_after) > 0
    assert pe_list_after[0][4] == 1, "Should have 1 selected target"
    print(f"Selected Targets: {pe_list_after[0][4]}")

    # 7. Generate Actions again - Should now offer RESOLVE_EFFECT
    actions_resolve = dm_ai_module.ActionGenerator.generate_legal_actions(game.state, card_db)
    resolve_action = None
    for a in actions_resolve:
        if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT:
            resolve_action = a
            break

    assert resolve_action is not None, "Should offer RESOLVE_EFFECT after selection is complete"

    # 8. Execute Resolve
    print("Executing Resolve...")
    dm_ai_module.EffectResolver.resolve_action(game.state, resolve_action, card_db)

    # 9. Verify Effect (Creature should be Tapped)
    opp_creature = game.state.players[1].battle_zone[0]
    print(f"Final Tapped State: {opp_creature.is_tapped}")
    assert opp_creature.is_tapped == True, "Creature should be tapped after spell resolution"

    # 10. Verify Pending Effect is gone
    pe_list_final = dm_ai_module.get_pending_effects_verbose(game.state)
    assert len(pe_list_final) == 0, "Pending effect should be gone"

if __name__ == "__main__":
    test_generic_targeting_tap()
