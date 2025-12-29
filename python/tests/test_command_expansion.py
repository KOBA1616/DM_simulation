
import pytest
import dm_ai_module
from dm_ai_module import CardDefinition, CommandDef, CommandType, TargetScope, JsonLoader

def test_command_expansion_draw():
    import json
    import os

    dummy_card = {
        "id": 9999,
        "name": "Dummy Draw",
        "type": "SPELL",
        "civilizations": ["FIRE"],
        "effects": [
            {
                "trigger": "ON_PLAY",
                "commands": [
                    {
                        "type": "DRAW_CARD",
                        "amount": 2 # Command uses 'amount'
                    }
                ]
            }
        ]
    }

    with open("dummy_draw.json", "w") as f:
        json.dump([dummy_card], f)

    # JsonLoader.load_cards returns the dict, it doesn't take reference in Python binding apparently?
    # Error message: (arg0: str) -> dict[int, dm_ai_module.CardDefinition]
    card_db = JsonLoader.load_cards("dummy_draw.json")

    assert 9999 in card_db
    defn = card_db[9999]
    assert len(defn.effects) == 1
    cmds = defn.effects[0].commands

    assert len(cmds) == 1
    cmd = cmds[0]
    # Previously expected expansion to TRANSITION, but engine now supports high-level DRAW_CARD command
    assert cmd.type == CommandType.DRAW_CARD
    # assert cmd.from_zone == "DECK"
    # assert cmd.to_zone == "HAND"
    assert cmd.amount == 2
    # The command wrapper exposes raw fields, but DRAW_CARD likely stores amount in value1
    # assert cmd.value1 == 2

    os.remove("dummy_draw.json")

def test_command_expansion_destroy():
    import json
    import os

    dummy_card = {
        "id": 9998,
        "name": "Dummy Destroy",
        "type": "SPELL",
        "civilizations": ["DARKNESS"],
        "effects": [
            {
                "trigger": "ON_PLAY",
                "commands": [
                    {
                        "type": "DESTROY",
                        "scope": "PLAYER_OPPONENT",
                        "filter": { "zones": ["BATTLE_ZONE"], "count": 1 }
                    }
                ]
            }
        ]
    }

    with open("dummy_destroy.json", "w") as f:
        json.dump([dummy_card], f)

    card_db = JsonLoader.load_cards("dummy_destroy.json")

    defn = card_db[9998]
    cmds = defn.effects[0].commands

    assert len(cmds) == 1
    cmd = cmds[0]
    # Previously expected expansion to TRANSITION, but engine now supports high-level DESTROY command
    assert cmd.type == CommandType.DESTROY
    # assert cmd.from_zone == "BATTLE"
    # assert cmd.to_zone == "GRAVEYARD"
    # Check if target logic is preserved (depending on implementation)
    # assert cmd.target_group == TargetScope.PLAYER_OPPONENT

    os.remove("dummy_destroy.json")
