
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
                "actions": [
                    {
                        "type": "DRAW_CARD",
                        "value1": 2 # Amount
                    }
                ]
            }
        ]
    }

    with open("dummy_draw.json", "w") as f:
        json.dump([dummy_card], f)

    card_db = JsonLoader.load_cards("dummy_draw.json")

    assert 9999 in card_db
    defn = card_db[9999]
    assert len(defn.effects) == 1
    cmds = defn.effects[0].commands

    assert len(cmds) == 1
    cmd = cmds[0]

    # Updated: DRAW_CARD legacy action maps to DRAW_CARD command (not TRANSITION)
    assert cmd.type == CommandType.DRAW_CARD
    # DRAW_CARD command uses 'amount' instead of explicit zones often, or implicitly checks owner
    # But let's check what was converted.
    assert cmd.amount == 2

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
                "actions": [
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

    # Updated: DESTROY maps to DESTROY command
    assert cmd.type == CommandType.DESTROY
    assert cmd.target_group == TargetScope.PLAYER_OPPONENT

    os.remove("dummy_destroy.json")
