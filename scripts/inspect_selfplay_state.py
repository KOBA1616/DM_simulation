import random
import time
import argparse
from dm_toolkit.gui.headless import create_session
from dm_toolkit import commands
# Prefer command-first wrapper
generate_legal_commands = commands.generate_legal_commands
import dm_ai_module


def inspect_game(max_steps=500, seed=42):
    random.seed(seed)
    sess = create_session()
    gs = sess.gs
    card_db = sess.card_db
    play_count = 0
    steps = 0
    attack_phase_seen = 0

    def dump_battle_zone(player_idx):
        try:
            p = gs.players[player_idx]
        except Exception:
            return []
        bz = getattr(p, 'battle_zone', []) or []
        out = []
        for c in bz:
            out.append({
                'instance_id': getattr(c, 'instance_id', None),
                'card_id': getattr(c, 'card_id', None),
                'is_tapped': getattr(c, 'is_tapped', None),
                'sick': getattr(c, 'sick', None),
            })
        return out

    start = time.time()
    while steps < max_steps and not sess.is_game_over():
        try:
            cmds = commands.generate_legal_commands(gs, card_db, strict=False, skip_wrapper=True) or []
        except Exception:
            try:
                cmds = commands.generate_legal_commands(gs, card_db) or []
            except Exception:
                cmds = []

        # Log phase transitions
        try:
            phase = getattr(gs, 'current_phase', None)
            if phase is None:
                phase = getattr(gs, 'phase', None)
            # try to get name if enum
            try:
                phase_name = phase.name
            except Exception:
                phase_name = str(phase)
        except Exception:
            phase_name = 'UNKNOWN'

        if 'ATTACK' in str(phase_name).upper():
            attack_phase_seen += 1
            ap = getattr(gs, 'active_player_id', None)
            ap = ap if ap is not None else getattr(gs, 'active_player', None)
            print(f"Step {steps}: ATTACK phase for player={ap}, battle_zone={dump_battle_zone(ap)}")

        # choose a command similar to existing selfplay
        choice = None
        if cmds:
            for c in cmds:
                try:
                    d = c.to_dict()
                except Exception:
                    d = {}
                t = d.get('type', '')
                if t and t != 'PASS':
                    choice = c
                    break
            if choice is None:
                choice = random.choice(cmds)

        if choice is not None:
            # Execute and log if PLAY_FROM_ZONE
            try:
                t = ''
                try:
                    t = choice.to_dict().get('type', '')
                except Exception:
                    pass
                sess.execute_action(choice)
                if t == 'PLAY_FROM_ZONE':
                    play_count += 1
                    ap = getattr(gs, 'active_player_id', None)
                    ap = ap if ap is not None else getattr(gs, 'active_player', None)
                    print(f"Step {steps}: PLAY_FROM_ZONE executed by player={ap}; battle_zone now={dump_battle_zone(ap)}")
            except Exception as e:
                print('execute_action raised', e)

        # advance phase
        try:
            sess.step_phase()
        except Exception:
            pass

        steps += 1

    duration = time.time() - start
    summary = {
        'steps': steps,
        'duration_s': duration,
        'play_from_zone_count': play_count,
        'attack_phase_seen': attack_phase_seen,
    }
    print('SUMMARY:', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    inspect_game(max_steps=args.steps, seed=args.seed)
