import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import dm_ai_module
from dm_toolkit import commands
print('imported dm_ai_module, IS_NATIVE=', getattr(dm_ai_module, 'IS_NATIVE', False))
try:
    gi = dm_ai_module.GameInstance(0)
    print('created GameInstance with 0')
except TypeError:
    gi = dm_ai_module.GameInstance()
    print('created GameInstance no-arg')
state = gi.state
print('got state')
try:
    if hasattr(state, 'initialize') and callable(state.initialize):
        state.initialize()
        print('initialized state')
except Exception as e:
    print('initialize failed', e)
print('setting active_player_id and phase')
state.active_player_id = 0
try:
    state.current_phase = dm_ai_module.Phase.MAIN
    print('set phase to Phase.MAIN')
except Exception:
    state.current_phase = 3
    print('set phase to 3')
print('adding card to hand')
try:
    state.add_card_to_hand(0, 1)
    print('add_card_to_hand(0,1) ok')
except TypeError:
    state.add_card_to_hand(0, 1, -1)
    print('add_card_to_hand(0,1,-1) ok')
print('adding card to mana')
try:
    state.add_card_to_mana(0, 1)
    print('add_card_to_mana ok')
except TypeError:
    state.add_card_to_mana(0, 1, -1)
    print('add_card_to_mana with iid ok')
try:
    for m in state.players[0].mana_zone:
        try:
            m.is_tapped = False
        except Exception:
            pass
    print('untapped mana')
except Exception as e:
    print('untap failed', e)
print('calling generate_legal_commands')
try:
    card_db = {1: {"id":1, "cost":1}, 99: {"id":99, "cost":99}}
    cmds = commands.generate_legal_commands(state, card_db)
    print('generate_legal_commands returned', len(cmds), 'commands')
    for c in cmds[:10]:
        try:
            print('cmd', c.to_string())
        except Exception:
            print('cmd repr', c)
except Exception as e:
    print('generate_legal_commands raised exception', e)
print('done')
