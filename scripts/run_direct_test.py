import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module
from dm_toolkit import commands


def run():
    try:
        gi = dm_ai_module.GameInstance(0)
    except TypeError:
        gi = dm_ai_module.GameInstance()
    state = gi.state
    try:
        state.initialize()
    except Exception:
        pass

    card_db = {
        1: {"id": 1, "name": "Cheap Creature", "cost": 1, "type": "CREATURE"},
        99: {"id": 99, "name": "Expensive", "cost": 99, "type": "CREATURE"}
    }

    state.active_player_id = 0
    try:
        state.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        state.current_phase = 3

    try:
        state.add_card_to_hand(0, 1)
    except TypeError:
        state.add_card_to_hand(0, 1, -1)

    try:
        state.add_card_to_mana(0, 1)
    except TypeError:
        state.add_card_to_mana(0, 1, -1)

    try:
        for m in state.players[0].mana_zone:
            m.is_tapped = False
    except Exception:
        pass

    cmds = []
    try:
        try:
            cmds = commands.generate_legal_commands(state, card_db, strict=False) or []
        except TypeError:
            cmds = commands.generate_legal_commands(state, card_db) or []
        except Exception:
            cmds = []
    except Exception:
        cmds = []
    if not cmds:
        try:
            cmds = commands.generate_legal_commands(state, card_db) or []
        except Exception:
            cmds = []
    print('Generated', len(cmds), 'commands')
    for c in cmds:
        try:
            print('->', c.to_dict())
        except Exception:
            try:
                print('->', c.to_string())
            except Exception:
                print('->', repr(c))

if __name__ == '__main__':
    run()
