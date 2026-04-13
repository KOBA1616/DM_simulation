from __future__ import annotations

import pytest

dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")


def _is_native() -> bool:
    return bool(getattr(dm_ai_module, "IS_NATIVE", False))


pytestmark = pytest.mark.skipif(
    not _is_native(), reason="Requires native dm_ai_module (IS_NATIVE=True)"
)


def _make_game():  # type: ignore[return]
    db = dm_ai_module.CardDatabase()
    game = dm_ai_module.GameInstance(7, db)
    game.start_game()
    return game, db


def _to_mana_phase(game, db) -> None:
    # START -> DRAW -> MANA
    for _ in range(2):
        dm_ai_module.PhaseManager.next_phase(game.state, db)


def test_pass_advances_single_phase_only() -> None:
    game, db = _make_game()
    _to_mana_phase(game, db)

    phase_before = int(game.state.current_phase)
    turn_before = int(game.state.turn_number)

    legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
    pass_cmd = None
    for cmd in legal:
        if "PASS" in str(cmd.type):
            pass_cmd = cmd
            break

    assert pass_cmd is not None, "PASS command not generated in MANA phase"

    game.resolve_command(pass_cmd)

    phase_after = int(game.state.current_phase)
    turn_after = int(game.state.turn_number)

    # 再発防止: resolve_command_oneshot/resolve_command の呼び出し後に
    # 呼び出し側が next_phase を重ねると MANA->ATTACK 等の二重進行が起きうる。
    assert phase_before == int(dm_ai_module.Phase.MANA)
    assert phase_after == int(dm_ai_module.Phase.MAIN)
    assert turn_after == turn_before
