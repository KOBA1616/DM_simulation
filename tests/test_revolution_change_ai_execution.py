# -*- coding: utf-8 -*-
"""革命チェンジの AI 実行検証。

再発防止:
- 攻撃時トリガー（ON_ATTACK 系）から革命チェンジの USE_ABILITY が合法手に出ること。
- SimpleAI が PASS より USE_ABILITY を優先して選び、実際に解決できること。
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple

import pytest


dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")
if not getattr(dm_ai_module, "IS_NATIVE", False):
    pytest.skip("Requires native dm_ai_module (IS_NATIVE=True)", allow_module_level=True)


_CARDS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "cards.json")
_MANA_INSTANCE_START = 9800


def _make_db() -> Any:
    if os.path.exists(_CARDS_JSON):
        return dm_ai_module.JsonLoader.load_cards(_CARDS_JSON)
    pytest.skip("cards.json が存在しない環境")


def _setup_game_for_card(card_id: int, mana_cost: int, seed: int = 42) -> Tuple[Any, Any]:
    db = _make_db()
    game = dm_ai_module.GameInstance(seed, db)
    s = game.state
    s.set_deck(0, [card_id] * 40)
    s.set_deck(1, [1] * 40)
    dm_ai_module.PhaseManager.start_game(s, db)

    for i in range(mana_cost):
        s.add_card_to_mana(0, card_id, _MANA_INSTANCE_START + i)

    for _ in range(30):
        ph = str(s.current_phase).upper()
        if "MAIN" in ph and s.active_player_id == 0:
            break
        legal: List[Any] = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if legal:
            game.resolve_command(legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)
    else:
        pytest.fail("MainPhase (P0) に到達できませんでした")

    return game, db


def _find_cmd_by_keyword(game: Any, db: Any, keyword: str) -> Any:
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
    for c in legal:
        if keyword.upper() in str(getattr(c, "type", "")).upper():
            return c
    return None


def _zone_instance_ids(zone_cards: List[Any]) -> List[int]:
    out: List[int] = []
    for c in zone_cards:
        iid = int(getattr(c, "instance_id", -1))
        if iid >= 0:
            out.append(iid)
    return out


def test_ai_selects_and_executes_revolution_change_use_ability() -> None:
    """AI が革命チェンジの USE_ABILITY を選び、解決できることを検証する。"""
    card_id = 3  # 芸魔王将カクメイジン (革命チェンジ持ち)
    game, db = _setup_game_for_card(card_id, mana_cost=7)
    s = game.state

    play_cmd = _find_cmd_by_keyword(game, db, "PLAY")
    if play_cmd is None:
        pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")
    game.resolve_command(play_cmd)

    attack_cmd = None
    for _ in range(80):
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        attack_cmd = next(
            (c for c in legal if "ATTACK_PLAYER" in str(getattr(c, "type", "")).upper()),
            None,
        )
        if attack_cmd is not None:
            break
        if legal:
            game.resolve_command(legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)

    if attack_cmd is None:
        pytest.skip("ATTACK_PLAYER コマンドが見つからず、革命チェンジ検証に進めません")

    game.resolve_command(attack_cmd)

    rc_legal: List[Any] = []
    for _ in range(20):
        rc_legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if any("USE_ABILITY" in str(getattr(c, "type", "")).upper() for c in rc_legal):
            break
        if rc_legal:
            game.resolve_command(rc_legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)

    use_cmds = [c for c in rc_legal if "USE_ABILITY" in str(getattr(c, "type", "")).upper()]
    assert use_cmds, (
        "革命チェンジの USE_ABILITY が合法手に生成されませんでした。\n"
        "再発防止: ON_ATTACK 系トリガーから ON_ATTACK_FROM_HAND 互換発火と"
        " PendingEffectStrategy の USE_ABILITY 生成を確認してください。"
    )

    ai = dm_ai_module.SimpleAI()
    selected_idx = ai.select_action(rc_legal, s)
    selected_cmd = rc_legal[selected_idx]
    assert "USE_ABILITY" in str(getattr(selected_cmd, "type", "")).upper(), (
        f"AI が革命チェンジを選択しませんでした: selected={selected_cmd.type}"
    )

    source_iid = int(getattr(selected_cmd, "instance_id", -1))
    target_iid = int(getattr(selected_cmd, "target_instance", -1))
    assert source_iid >= 0 and target_iid >= 0

    game.resolve_command(selected_cmd)

    p0 = s.players[0]
    hand_ids = _zone_instance_ids(p0.hand)
    battle_ids = _zone_instance_ids(p0.battle_zone)

    assert source_iid in battle_ids, "革命チェンジで出る側のカードがバトルゾーンに存在しません"
    assert target_iid in hand_ids, "革命チェンジで戻る側のカードが手札に存在しません"
