from __future__ import annotations

import pytest

dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")


def _is_native() -> bool:
    return bool(getattr(dm_ai_module, "IS_NATIVE", False))


pytestmark = pytest.mark.skipif(
    not _is_native(), reason="Requires native dm_ai_module (IS_NATIVE=True)"
)


def _make_game(seed: int):  # type: ignore[return]
    db = dm_ai_module.CardDatabase()
    game = dm_ai_module.GameInstance(seed, db)
    game.start_game()
    return game


def test_step_with_reason_deterministic_same_seed() -> None:
    if not hasattr(dm_ai_module.GameInstance, "step_with_reason"):
        pytest.skip("step_with_reason is not available in this native build")

    seed = 12345
    g1 = _make_game(seed)
    g2 = _make_game(seed)

    reasons1: list[str] = []
    reasons2: list[str] = []

    for _ in range(120):
        r1 = str(g1.step_with_reason())
        r2 = str(g2.step_with_reason())
        reasons1.append(r1)
        reasons2.append(r2)

        if "GAME_OVER" in r1 and "GAME_OVER" in r2:
            break

    assert reasons1 == reasons2


def test_step_bool_compat_matches_reason_executed() -> None:
    if not hasattr(dm_ai_module.GameInstance, "step_with_reason"):
        pytest.skip("step_with_reason is not available in this native build")

    seed = 24680
    g_reason = _make_game(seed)
    g_bool = _make_game(seed)

    for _ in range(120):
        reason = str(g_reason.step_with_reason())
        result = bool(g_bool.step())

        # 再発防止: bool 互換 API と理由 API の判定差分が出ると
        # Python 側ワーカーの停止判定が経路ごとに食い違うため一致を固定する。
        assert ("EXECUTED" in reason) == result

        if "GAME_OVER" in reason:
            break