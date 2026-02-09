import traceback
import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'bin', 'Release'))
print('Starting manual test run')
try:
    import dm_ai_module
    from dm_toolkit import commands
    print('Imports ok')

    try:
        gi = dm_ai_module.GameInstance(0)
    except TypeError:
        gi = dm_ai_module.GameInstance()
    state = gi.state
    print('GameInstance created')

    if hasattr(state, 'initialize') and callable(state.initialize):
        try:
            state.initialize()
            print('state.initialize called')
        except Exception:
            print('state.initialize raised')

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

    print('Calling commands.generate_legal_commands (prefer command-first)')
    try:
        cmds = commands.generate_legal_commands(state, card_db, strict=False) or []
    except Exception:
        try:
            cmds = commands.generate_legal_commands(state, card_db) or []
        except Exception:
            cmds = []
    print(f'generate_legal_commands returned {len(cmds)} commands')

    found_play = False
    for c in cmds:
        try:
            d = c.to_dict()
            if d.get('unified_type') == 'PLAY' or d.get('type') == 'PLAY_FROM_ZONE':
                found_play = True
                break
        except Exception:
            continue

    print('Found play candidate:', found_play)
    if not found_play:
        print('Commands:', [getattr(c, 'to_string', lambda: str(c))() for c in cmds])

except Exception:
    print('EXCEPTION during manual test:')
    traceback.print_exc()
    raise
else:
    print('Manual test finished')
