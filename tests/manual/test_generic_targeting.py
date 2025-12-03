
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

    card_db = {
        9000: dm_ai_module.CardDefinition(
            id=9000,
            name="Tap Spell",
            civilization="LIGHT",
            races=["Spell"],
            cost=1,
            power=0,
            keywords=dm_ai_module.CardKeywords(),
            effects=[
                dm_ai_module.EffectDef(
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
            ]
        ),
        9001: dm_ai_module.CardDefinition(
            id=9001,
            name="Dummy Creature",
            civilization="FIRE",
            races=["Human"],
            cost=2,
            power=2000,
            keywords=dm_ai_module.CardKeywords(),
            effects=[]
        )
    }

    # 2. Setup Game
    # Use GameInstance constructor with card_db if available, or just create empty and set state
    # But current binding: .def(py::init([]() { return new GameInstance(0, empty_db); }))
    # And .def("start_game", ...)

    game = dm_ai_module.GameInstance()
    # game.initialize(card_db) # Doesn't exist
    game.start_game(card_db)   # Exists in binding

    # Force Main Phase for testing
    game.state.current_phase = dm_ai_module.Phase.MAIN
    game.state.active_player_id = 0

    # Player 0 has the Spell
    game.state.add_card_to_hand(0, 9000, 100)
    game.state.add_card_to_mana(0, 9000, 101) # Mana source

    # Player 1 has a Creature (Untapped)
    game.state.add_card_to_battle(1, 9001, 200)
    game.state.players[1].battle_zone[0].is_tapped = False
    game.state.players[1].battle_zone[0].summoning_sickness = False

    # 3. Player 0 Plays the Spell
    # Main Phase -> Generate Actions -> Play Card 9000
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game.state, card_db)
    play_action = None
    for a in actions:
        if a.type == dm_ai_module.ActionType.PLAY_CARD and a.card_id == 9000:
            play_action = a
            break

    assert play_action is not None, "Should be able to play the spell"
    # game.step(play_action) -> We don't have step() in C++ GameInstance exposed directly maybe?
    # We have dm_ai_module.EffectResolver.resolve_action usually or PhaseManager logic.
    # But usually tests use a wrapper.
    # We can use dm_ai_module.EffectResolver.resolve_action(game.state, play_action, card_db) (Wait, EffectResolver takes Action)
    # The signature in binding: EffectResolver.resolve_action(state, action) ?? No, check binding
    # Binding: .def_static("resolve_action", &EffectResolver::resolve_action);
    # C++: void EffectResolver::resolve_action(GameState&, const Action&, const map& card_db)
    # So we need to pass card_db.

    # Wait, EffectResolver binding in bindings.cpp:
    # .def_static("resolve_action", &EffectResolver::resolve_action);
    # This might fail if python doesn't match arguments.

    # Let's check bindings.cpp again.
    # It just exposes resolve_action.

    dm_ai_module.EffectResolver.resolve_action(game.state, play_action, card_db)

    # 4. Expect Pending Effect (Target Selection)
    # The engine should pause and ask for target selection.
    # Current phase might still be MAIN, but legal actions should be SELECT_TARGET

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

    # 6. Verify Effect (Creature should be Tapped)
    opp_creature = game.state.players[1].battle_zone[0]
    assert opp_creature.is_tapped == True, "Creature should be tapped after spell resolution"

    # 7. Game should continue (e.g. back to Main Phase or resolved)
    # Spell should be in graveyard? (For spells, ON_PLAY usually sends to GY after?)
    # GenericCardSystem::MEKRAID handles spells specially, but normal PLAY_CARD
    # usually puts spells in GY *after* effect?
    # Or EffectResolver puts it in GY.

    # Verify spell is in graveyard
    assert len(game.state.players[0].graveyard) > 0
    assert game.state.players[0].graveyard[0].card_id == 9000

if __name__ == "__main__":
    test_generic_targeting_tap()
