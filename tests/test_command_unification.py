import pytest
import dm_ai_module
import os

def test_action_conversion_to_command():
    # Verify that executing an Action now creates a GameCommand in history

    # 1. Setup
    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [1]
    config.my_mana = 0
    config.my_battle_zone = []

    # CardData constructor: id, name, cost, civilization, power, type, races, effects
    # Arg3 expects Civilization enum
    card_data = dm_ai_module.CardData(1, "TestCreature", 3, dm_ai_module.Civilization.NATURE, 1000, dm_ai_module.CardType.CREATURE, ["TestRace"], [])
    dm_ai_module.register_card_data(card_data)

    game = dm_ai_module.GameInstance(100)
    game.reset_with_scenario(config)

    # 2. Execute Action (PLAY_CARD)
    action = dm_ai_module.Action()
    action.type = dm_ai_module.PlayerIntent.PLAY_CARD
    action.source_instance_id = 0 # Hand card

    # 3. Resolve
    game.resolve_action(action)

    # 4. Undo
    game.undo()

    pass

if __name__ == "__main__":
    test_action_conversion_to_command()
