import sys, os, random
sys.path.insert(0, os.getcwd())
if os.path.isdir('python'):
    sys.path.insert(0, os.path.abspath('python'))
import dm_ai_module
from dm_toolkit import commands_v2 as commands
from dm_toolkit.engine.compat import EngineCompat

card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
valid_ids = list(card_db.keys())
if not valid_ids:
    print('no cards')
    raise SystemExit(1)

def replay(seed, deck=None, max_steps=200):
    deck = deck or [random.choice(valid_ids) for _ in range(40)]
    inst = dm_ai_module.GameInstance(seed, card_db)
    gs = inst.state
    gs.set_deck(0, deck)
    gs.set_deck(1, deck)
    try:
        inst.start_game()
    except Exception:
        dm_ai_module.PhaseManager.start_game(gs, card_db)

    print(f"Start replay seed={seed}")
    print(f"Initial: turn={gs.turn_number} phase={gs.current_phase} active={gs.active_player_id}")

    try:
        Heur = dm_ai_module.HeuristicAgent
        ag0 = Heur(0, card_db)
        ag1 = Heur(1, card_db)
    except Exception:
        ag0 = None
        ag1 = None

    for step in range(max_steps):
        print(f"STEP {step}: turn={gs.turn_number} phase={gs.current_phase} active={gs.active_player_id} winner={gs.winner}")
        print(f"  p0 hand={len(gs.players[0].hand)} p1 hand={len(gs.players[1].hand)} deck0={len(gs.players[0].deck)} deck1={len(gs.players[1].deck)}")
        try:
            legal = commands.generate_legal_commands(gs, card_db, strict=False) or []
        except Exception:
            try:
                legal = commands.generate_legal_commands(gs, card_db) or []
            except Exception:
                legal = []
        print(f"  legal_actions={len(legal) if legal is not None else 'None'}")
        if gs.winner != dm_ai_module.GameResult.NONE:
            print('Winner set:', gs.winner)
            break
        if not legal:
            print('No legal actions -> advancing phases')
            ff = 0
            while True:
                if gs.winner != dm_ai_module.GameResult.NONE:
                    break
                try:
                    legal2 = commands.generate_legal_commands(gs, card_db, strict=False) or []
                except Exception:
                    try:
                        legal2 = commands.generate_legal_commands(gs, card_db) or []
                    except Exception:
                        legal2 = []
                if not legal2:
                    try:
                        from dm_toolkit import commands as legacy_commands
                        legal2 = legacy_commands._call_native_action_generator(gs, card_db) or []
                    except Exception:
                        legal2 = []
                if legal2:
                    break
                dm_ai_module.PhaseManager.next_phase(gs, card_db)
                ff += 1
                if ff > 20:
                    print('  fast-forward limit reached')
                    break
            continue
        if gs.active_player_id == 0:
            action = ag0.get_action(gs, legal) if ag0 else None
        else:
            action = ag1.get_action(gs, legal) if ag1 else None
        print('  Chosen action type:', getattr(action, 'type', None))

        try:
            from dm_toolkit.compat_wrappers import execute_action_compat
            execute_action_compat(inst.state, action, card_db)
        except Exception as e:
            print('  resolve_action exception:', e)
            try:
                dm_ai_module.GameLogicSystem.resolve_action_oneshot(gs, action, card_db)
            except Exception as e2:
                print('  fallback resolve exception:', e2)
        # after action
        # after action
    print('Replay finished')

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 791488495
    replay(seed)
