# -*- coding: utf-8 -*-
"""Python フォールバック generate_legal_commands テスト。

再発防止:
  - このテストは DM_DISABLE_NATIVE=1 で Python フォールバックパスのみを検証する。
    ネイティブパスは test_command_flow.py / test_game_integrity.py で検証する。
  - GameInstance(seed) のみ (db なし) は TypeError になる。
    必ず JsonLoader.load_cards() → GameInstance(seed, db) の順で初期化すること。
  - add_card_to_hand / add_card_to_mana は常に (player_idx, card_id, instance_id) の
    3引数形式を使うこと。2引数形式は旧 API であり削除済み。
"""
from __future__ import annotations
import os
import pytest

dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires dm_ai_module")

from dm_ai_module import JsonLoader
from dm_toolkit import commands


def test_generate_play_candidates_present() -> None:
    """DM_DISABLE_NATIVE=1 (Python フォールバック) 時に手札のカードが
    PLAY_FROM_ZONE コマンドとして返ることを確認する。

    再発防止: commands.py の DM_DISABLE_NATIVE フォールバックは
              state.players[pid].hand の各カードを PLAY_FROM_ZONE に変換する。
              この動作が壊れていないことをここで回帰テストする。
    """
    # 再発防止: GameInstance(seed) のみは TypeError。必ず db を渡すこと。
    _cards_path = "data/cards.json"
    db = JsonLoader.load_cards(_cards_path) if os.path.exists(_cards_path) else dm_ai_module.CardDatabase()
    gi = dm_ai_module.GameInstance(0, db)
    state = gi.state

    # MAIN フェーズ・アクティブプレイヤー 0 にセット
    state.active_player_id = 0
    state.current_phase = dm_ai_module.Phase.MAIN

    # 再発防止: add_card_to_hand / add_card_to_mana は(player_idx, card_id, instance_id)の3引数
    state.add_card_to_hand(0, 1, 100)
    state.add_card_to_mana(0, 1, 101)

    # Python フォールバック用ダミー card_db（コスト判定に使用）
    card_db = {1: {"id": 1, "name": "Test Creature", "cost": 1, "type": "CREATURE"}}

    # DM_DISABLE_NATIVE=1 で Python フォールバックを強制（C++ 生成パスをスキップ）
    _prev = os.environ.get("DM_DISABLE_NATIVE")
    try:
        os.environ["DM_DISABLE_NATIVE"] = "1"
        cmds = commands.generate_legal_commands(state, card_db)
    finally:
        if _prev is None:
            os.environ.pop("DM_DISABLE_NATIVE", None)
        else:
            os.environ["DM_DISABLE_NATIVE"] = _prev

    found_play = False
    for c in cmds:
        try:
            d = c.to_dict() if hasattr(c, "to_dict") else c
            if isinstance(d, dict):
                t = d.get("type")
                # 再発防止: ネイティブは CommandType enum、フォールバックは文字列なので両方チェック
                type_name = t.name if hasattr(t, "name") else str(t)
                if d.get("unified_type") == "PLAY" or type_name == "PLAY_FROM_ZONE":
                    found_play = True
                    break
        except Exception:
            continue

    assert found_play, (
        f"PLAY_FROM_ZONE が Python フォールバックで生成されなかった: {cmds}\n"
        "再発防止: commands.py の DM_DISABLE_NATIVE フォールバックで手札の各カードを"
        " PLAY_FROM_ZONE に変換していることを確認すること。"
    )
