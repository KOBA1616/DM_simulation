from dm_toolkit.gui.headless import create_session
import dm_ai_module


def make_creature(card_id=1, iid_start=2000):
    class C:
        pass
    c = C()
    c.card_id = card_id
    c.instance_id = iid_start
    c.is_tapped = False
    c.sick = False
    return c


if __name__ == '__main__':
    sess = create_session()
    gs = sess.gs
    card_db = sess.card_db
    # Ensure active player is 0 and phase is ATTACK
    try:
        gs.active_player_id = 0
    except Exception:
        try:
            gs.active_player = 0
        except Exception:
            pass
    try:
        gs.current_phase = dm_ai_module.Phase.ATTACK
    except Exception:
        gs.current_phase = 4

    # Place an untapped, non-sick creature for active player
    try:
        p = gs.players[0]
    except Exception:
        raise SystemExit('No players in gs')

    # Ensure battle_zone exists
    if not hasattr(p, 'battle_zone'):
        p.battle_zone = []

    # Add a creature instance
    cre = make_creature(card_id=1, iid_start=3001)
    p.battle_zone.append(cre)

    # Prefer command-first generator (compat wrapper)
    from dm_toolkit.commands import generate_legal_commands
    cmds = generate_legal_commands(gs, card_db)
    print('Wrapped cmds count:', len(cmds))
    for i, c in enumerate(cmds[:20]):
        try:
            print('CMD', i, c.to_dict())
        except Exception:
            try:
                print('CMD', i, repr(c))
            except Exception:
                print('CMD', i, str(c))
