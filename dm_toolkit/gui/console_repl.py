"""Headless main-window simulator (REPL).

Provides a small REPL to inspect and control a GameSession similarly to the GUI
main window, but without importing Qt. Useful for debugging and CI.
"""
import sys
import time
import json
import logging
from scripts.logging_manager import configure_logging, get_logger
from typing import Any

from dm_toolkit.gui import headless

def show_status(sess: Any):
    gs = getattr(sess, 'gs', None)
    active = None
    phase = None
    try:
        active = getattr(gs, 'active_player_id', None)
    except Exception:
        active = None
    try:
        phase = getattr(gs, 'phase', None) or getattr(gs, 'current_phase', None)
    except Exception:
        phase = None

    print(f"Session: gs_present={'yes' if gs else 'no'} active_player={active} phase={phase}")


def list_hand(sess: Any, pid: int = 0):
    gs = getattr(sess, 'gs', None)
    if not gs:
        print("no game state")
        return
    try:
        p = gs.players[pid]
    except Exception:
        print(f"no player {pid}")
        return
    hand = getattr(p, 'hand', []) or []
    if not hand:
        print("hand empty")
        return
    for c in hand:
        iid = getattr(c, 'instance_id', None)
        cid = getattr(c, 'card_id', None)
        print(f"iid={iid} card_id={cid} repr={c}")


def show_deck_sizes(sess: Any, pid: int = 0):
    gs = getattr(sess, 'gs', None)
    if not gs:
        print("no game state")
        return
    try:
        p = gs.players[pid]
    except Exception:
        print(f"no player {pid}")
        return
    d = getattr(p, 'deck', []) or []
    print(f"player {pid} deck_size={len(d)} hand_size={len(getattr(p,'hand',[]) or [])} mana_size={len(getattr(p,'mana_zone',[]) or [])}")


def list_legal(sess: Any):
    if not getattr(sess, 'gs', None):
        print("no game state")
        return
    from dm_toolkit import commands_v2
    try:
        import dm_ai_module
    except Exception:
        from dm_toolkit import dm_ai_module

    try:
        cmds = commands_v2.generate_legal_commands(sess.gs, sess.card_db, strict=False) or []
    except Exception as e:
        print("commands_v2.generate_legal_commands failed:", e)
        cmds = []

    if not cmds:
        try:
            from dm_toolkit import commands_v2 as _commands
            cmds = _commands.generate_legal_commands(sess.gs, sess.card_db) or []
        except Exception:
            try:
                cmds = commands_v2.generate_legal_commands(sess.gs, sess.card_db) or []
            except Exception as e:
                print("commands_v2.generate_legal_commands failed:", e)
                return
    out = []
    for i, c in enumerate(cmds):
        try:
            d = c.to_dict()
        except Exception:
            d = {'type': str(type(c))}
        print(f"[{i}] {d}")
    print(f"total legal commands: {len(cmds)}")
    return cmds


def play_instance(sess: Any, iid: int) -> bool:
    ok = headless.play_instance(sess, iid)
    print("played" if ok else "no playable command for instance")
    return ok


def play_first(sess: Any) -> bool:
    gs = getattr(sess, 'gs', None)
    if not gs:
        print("no game state")
        return False
    p = gs.players[gs.active_player_id]
    hand = getattr(p, 'hand', []) or []
    for c in hand:
        iid = getattr(c, 'instance_id', None)
        if iid is None:
            continue
        if headless.play_instance(sess, iid):
            print(f"played instance {iid}")
            return True
    print("no playable hand instance")
    return False


def repl(sess: Any, auto_iterations: int = None):
    if auto_iterations is not None:
        for it in range(auto_iterations):
            print(f"--- auto iteration {it+1}/{auto_iterations} ---")
            played = play_first(sess)
            if not played:
                try:
                    sess.step_phase()
                except Exception:
                    pass
            time.sleep(0.01)
        return

    while True:
        try:
            cmd = input('> ').strip()
        except EOFError:
            break

        if not cmd:
            continue
        parts = cmd.split()
        key = parts[0]
        if key in ('exit', 'quit'):
            break
        if key == 'status':
            show_status(sess)
        elif key == 'hand':
            pid = int(parts[1]) if len(parts) > 1 else 0
            list_hand(sess, pid)
        elif key == 'deck':
            pid = int(parts[1]) if len(parts) > 1 else 0
            show_deck_sizes(sess, pid)
        elif key == 'legal':
            list_legal(sess)
        elif key == 'play':
            if len(parts) < 2:
                print('usage: play <instance_id>')
            else:
                try:
                    iid = int(parts[1])
                    play_instance(sess, iid)
                except Exception:
                    print('bad instance id')
        elif key == 'play-first':
            play_first(sess)
        elif key == 'step':
            sess.step_phase()
            print('stepped')
        elif key == 'run':
            n = int(parts[1]) if len(parts) > 1 else 10
            steps, go = headless.run_steps(sess, n)
            print(f'run steps={steps} game_over={go}')
        elif key == 'auto':
            n = int(parts[1]) if len(parts) > 1 else 10
            for i in range(n):
                played = play_first(sess)
                if not played:
                    sess.step_phase()
                time.sleep(0.01)
            print('auto finished')
        elif key == 'dump':
            gs = getattr(sess, 'gs', None)
            print('GS:', gs)
            if gs:
                for pid, p in enumerate(gs.players):
                    print(f'Player {pid}: deck={len(getattr(p,"deck",[]))} hand={len(getattr(p,"hand",[]))} mana={len(getattr(p,"mana_zone",[]))}')
        else:
            print('unknown command:', key)


def dump_minimal_state(sess: Any, path: str):
    """Write a minimal JSON snapshot of players' hands/decks to `path`.

    The snapshot contains for each player: deck_size, hand (list of {instance_id,card_id}).
    """
    gs = getattr(sess, 'gs', None)
    out = {'players': []}
    if not gs:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        return

    for pid, p in enumerate(gs.players):
        try:
            hand = getattr(p, 'hand', []) or []
            deck = getattr(p, 'deck', []) or []
            hand_items = []
            for c in hand:
                hand_items.append({'instance_id': getattr(c, 'instance_id', None), 'card_id': getattr(c, 'card_id', None)})
            out['players'].append({'player_id': pid, 'deck_size': len(deck), 'hand': hand_items})
        except Exception:
            out['players'].append({'player_id': pid, 'deck_size': None, 'hand': []})

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


def run_console(args):
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    # Delegate logging setup to centralized manager so CLI and GUI behave consistently
    configure_logging(console_level_name=args.log_level)
    logger = get_logger('headless_console')

    sess = headless.create_session(p0_human=args.p0_human, p1_human=args.p1_human)
    logger.info('Session created. gs present: %s', bool(getattr(sess, 'gs', None)))
    if args.auto is not None:
        repl(sess, auto_iterations=args.auto)
        logger.info('auto run complete')
        if args.dump_json:
            try:
                dump_minimal_state(sess, args.dump_json)
                logger.info('dumped JSON to %s', args.dump_json)
            except Exception as e:
                logger.warning('failed to dump json: %s', e)
        return

    repl(sess)
    if args.dump_json:
        try:
            dump_minimal_state(sess, args.dump_json)
            logger.info('dumped JSON to %s', args.dump_json)
        except Exception as e:
            logger.warning('failed to dump json: %s', e)
