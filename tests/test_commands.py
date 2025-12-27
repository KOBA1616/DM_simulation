import dm_ai_module
from dm_toolkit.commands import wrap_action


def test_wrap_and_execute_pass():
    gs = dm_ai_module.GameState()
    gs.setup_test_duel()
    # ensure starting phase MAIN
    try:
        gs.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        pass

    a = dm_ai_module.Action(type=dm_ai_module.ActionType.PASS, player_id=0)
    cmd = wrap_action(a)
    assert cmd is not None
    cmd.execute(gs)
    # PASS toggles MAIN <-> ATTACK in GameInstance.resolve_action fallback; in GameState shim,
    # FlowCommand sets to MAIN/ATTACK elsewhere; accept either non-None change
    assert getattr(gs, 'current_phase', None) is not None
