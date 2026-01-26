import sys
import os
import traceback
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

sys.path.insert(0, os.getcwd())
if os.path.isdir("python"):
    sys.path.insert(0, os.path.abspath("python"))

from dm_toolkit.gui.headless import create_session, find_legal_commands_for_instance, play_instance, run_steps


def main():
    try:
        sess = create_session(p0_human=True)
        print("Session created. gs present:", bool(sess.gs))

        if not sess.gs:
            print("No GameState available; aborting smoke test.")
            return 2

        # attach simple logger
        def log_cb(m):
            try:
                print("LOG:", m)
            except Exception:
                pass

        sess.callback_log = log_cb
        sess.callback_update_ui = lambda: None

        # List P0 hand instance ids
        try:
            p0 = sess.gs.players[0]
            hand = getattr(p0, 'hand', [])
            hand_ids = [getattr(c, 'instance_id', getattr(c, 'id', None)) for c in hand]
        except Exception:
            hand_ids = []

        print("P0 hand instance ids:", hand_ids)

        if not hand_ids:
            print("Hand empty — running one step to populate/draw")
            sess.step_phase()
            try:
                p0 = sess.gs.players[0]
                hand = getattr(p0, 'hand', [])
                hand_ids = [getattr(c, 'instance_id', getattr(c, 'id', None)) for c in hand]
            except Exception:
                hand_ids = []

        print("Post-step P0 hand instance ids:", hand_ids)

        if hand_ids:
            iid = hand_ids[0]
            cmds = find_legal_commands_for_instance(sess, iid)
            print(f"Found {len(cmds)} legal commands for instance {iid}")
            for c in cmds[:5]:
                try:
                    print(' ->', c.to_dict())
                except Exception:
                    print(' ->', str(c))

            ok = play_instance(sess, iid)
            print('play_instance returned', ok)
        else:
            print('No instances to try play.')

        steps, over = run_steps(sess, max_steps=50)
        print('run_steps ->', steps, 'game_over=', over)

        return 0
    except Exception:
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    sys.exit(main())
