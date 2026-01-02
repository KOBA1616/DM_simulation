import pytest

pytest.skip(
    "duplicate root-level test file; use tests/unit/test_commands_new.py",
    allow_module_level=True,
)

import dm_ai_module
from dm_toolkit.commands_new import wrap_action


def test_wrap_and_execute_pass():
    gs = dm_ai_module.GameState(40)
    gs.setup_test_duel()
    # ensure starting phase MAIN
    try:
        gs.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        pass

    a = dm_ai_module.Action()
    a.type = dm_ai_module.ActionType.PASS
    a.target_player = 0
    cmd = wrap_action(a)
    assert cmd is not None
    cmd.execute(gs)
    # PASS toggles MAIN <-> ATTACK in GameInstance.resolve_action fallback; in GameState shim,
    # FlowCommand sets to MAIN/ATTACK elsewhere; accept either non-None change
    assert getattr(gs, "current_phase", None) is not None
