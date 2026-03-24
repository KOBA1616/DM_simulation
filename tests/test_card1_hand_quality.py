# -*- coding: utf-8 -*-
"""
TDD: 月光電人オボロカゲロウ (card id=1) 手札交換品質テスト
======================================================================
期待される正しい振る舞い:
  1. カードをプレイ → ON_PLAY 効果が発火
  2. QUERY(MANA_CIVILIZATION_COUNT) で上限 N を決定
  3. DRAW_CARD(up_to=True) → SELECT_NUMBER(0..N) でドロー枚数を選択
  4. SELECT_TARGET(HAND, count=N) → プレイヤーが「手札の中からどれを戻すか」を選択
  5. TRANSITION(chosen_targets → DECK_BOTTOM) → 選択したカードが山札底へ

バグ修正前の問題:
  ・ TRANSITION が SELECT_NUMBER の回答前に実行されるため var_DRAW_CARD_1=0 で空移動
  ・ SELECT_TARGET が生成されないため「どのカードを戻すか」選べない
  ・ ドローしたカードが手札に残り続け、手札は一方的に増える

再発防止:
  - EffectSystem は全コマンドを単一パイプラインで実行すること
  - QUERY(SELECT_TARGET) は InstructionOp::SELECT を生成し
    実行時に HAND の対象カードを動的に列挙すること
  - TRANSITION の input_value_key は vector<int> を受け取ること
"""
from __future__ import annotations
import os
import pathlib
from typing import Any, List, Optional

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

dm = pytest.importorskip("dm_ai_module", reason="Requires native engine")
if not getattr(dm, "IS_NATIVE", False):
    pytest.skip("Requires native dm_ai_module (IS_NATIVE=True)", allow_module_level=True)

_CARDS_JSON = str(_ROOT / "data" / "cards.json")
_MANA_START = 9200


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def _setup(card_id: int = 1, mana_cost: int = 2, seed: int = 42) -> tuple[Any, Any]:
    db = dm.JsonLoader.load_cards(_CARDS_JSON)
    game = dm.GameInstance(seed, db)
    s = game.state
    s.set_deck(0, [card_id] * 40)
    s.set_deck(1, [1] * 40)
    dm.PhaseManager.start_game(s, db)
    for i in range(mana_cost):
        s.add_card_to_mana(0, card_id, _MANA_START + i)
    for _ in range(30):
        if "MAIN" in str(s.current_phase).upper() and s.active_player_id == 0:
            break
        legal = dm.IntentGenerator.generate_legal_commands(s, db)
        if legal:
            game.resolve_command(legal[0])
        else:
            dm.PhaseManager.next_phase(s, db)
    return game, db


def _cmd(game: Any, db: Any, keyword: str) -> Optional[Any]:
    return next(
        (c for c in dm.IntentGenerator.generate_legal_commands(game.state, db)
         if keyword.upper() in str(getattr(c, "type", "")).upper()),
        None,
    )


def _all_cmds(game: Any, db: Any) -> List[Any]:
    return dm.IntentGenerator.generate_legal_commands(game.state, db)


def _instance_ids_in_zone(state: Any, player: int, zone: str) -> List[int]:
    """指定ゾーンのインスタンスID一覧を返す。"""
    p = state.players[player]
    if zone == "hand":
        return [c.instance_id for c in p.hand]
    if zone == "deck":
        return [c.instance_id for c in p.deck]
    if zone == "graveyard":
        return [c.instance_id for c in p.graveyard]
    return []


# ─────────────────────────────────────────────
# テストクラス
# ─────────────────────────────────────────────

class TestCard1HandQuality:
    """月光電人オボロカゲロウの手札交換品質テスト。"""

    CARD_ID = 1
    COST = 2

    # ------------------------------------------------------------------
    # Red: 現在失敗するテスト（実装が正しくなれば Green になる）
    # ------------------------------------------------------------------

    def test_select_target_appears_after_draw(self) -> None:
        """ドロー枚数選択(SELECT_NUMBER)の後に SELECT_TARGET が生成されること。

        手札から「どれを戻すか」を選ぶコマンドが生成されなければ
        手札品質の改善は起きていない。
        """
        game, db = _setup(self.CARD_ID, self.COST, seed=10)
        s = game.state

        play_cmd = _cmd(game, db, "PLAY")
        if play_cmd is None:
            pytest.skip("PLAY コマンドなし")
        game.resolve_command(play_cmd)

        resolve_cmd = _cmd(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドなし")
        game.resolve_command(resolve_cmd)

        # SELECT_NUMBER で 1 枚ドローを選択
        sel_num = _cmd(game, db, "SELECT_NUMBER")
        if sel_num is not None:
            choose_1 = next(
                (c for c in _all_cmds(game, db)
                 if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
                 and getattr(c, "target_instance", -1) == 1),
                sel_num,
            )
            game.resolve_command(choose_1)

        # ▶ 修正後: SELECT_TARGET が生成される
        # ▶ 修正前: 生成されず PASS しか来ない → このアサートが失敗する
        legal = _all_cmds(game, db)
        has_select_target = any(
            "SELECT_TARGET" in str(getattr(c, "type", "")).upper()
            for c in legal
        )
        assert has_select_target, (
            "SELECT_NUMBER 回答後に SELECT_TARGET が生成されていません。\n"
            f"  生成されたコマンド: {[str(getattr(c,'type','?')) for c in legal]}\n"
            "  再発防止:\n"
            "  1. cards.json card id=1 に QUERY(SELECT_TARGET, HAND) を追加\n"
            "  2. EffectSystem を単一パイプラインで実行するよう修正\n"
            "  3. QUERY SELECT_TARGET が InstructionOp::SELECT を生成するよう修正"
        )

    def test_specific_old_card_goes_to_deck_bottom(self) -> None:
        """プレイヤーが選択した特定の（元々の）手札カードが山札底へ移動すること。

        バグ修正前: ドローしたカード（または0枚）しか戻らない。
        バグ修正後: 指定したインスタンスIDが deck に移動している。
        """
        game, db = _setup(self.CARD_ID, self.COST, seed=20)
        s = game.state

        # プレイ前の手札の instance_id を記録
        hand_before_ids = set(_instance_ids_in_zone(s, 0, "hand"))

        play_cmd = _cmd(game, db, "PLAY")
        if play_cmd is None:
            pytest.skip("PLAY コマンドなし")
        game.resolve_command(play_cmd)

        # プレイ後（手札に残っている元のカード = 戻す候補）
        hand_after_play_ids = set(_instance_ids_in_zone(s, 0, "hand"))
        # 元々手札にあったカードで、プレイしたカード以外
        old_hand_ids = hand_before_ids & hand_after_play_ids

        resolve_cmd = _cmd(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドなし")
        game.resolve_command(resolve_cmd)

        # SELECT_NUMBER → 1 枚ドロー選択
        sel_num = _cmd(game, db, "SELECT_NUMBER")
        if sel_num is not None:
            choose_1 = next(
                (c for c in _all_cmds(game, db)
                 if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
                 and getattr(c, "target_instance", -1) == 1),
                sel_num,
            )
            game.resolve_command(choose_1)

        # SELECT_TARGET で「元の手札のカード」を1枚選んで戻す
        for _ in range(5):
            legal = _all_cmds(game, db)
            target_cmds = [
                c for c in legal
                if "SELECT_TARGET" in str(getattr(c, "type", "")).upper()
            ]
            if not target_cmds:
                break
            # old_hand_ids に含まれるインスタンスを優先して選択
            best = next(
                (c for c in target_cmds
                 if getattr(c, "instance_id", None) in old_hand_ids),
                target_cmds[0],
            )
            chosen_id = getattr(best, "instance_id", None)
            game.resolve_command(best)

            # ループ終了判定：待機解除
            if not s.waiting_for_user_input:
                break

        # ▶ 修正後: chosen_id が deck に移動している
        deck_ids = set(_instance_ids_in_zone(s, 0, "deck"))
        assert chosen_id in deck_ids, (
            f"選択したカード (instance_id={chosen_id}) が山札に見つかりません。\n"
            f"  山札 instance_ids (先頭5件): {list(deck_ids)[:5]}\n"
            "  再発防止:\n"
            "  - SELECT_TARGET の結果が TRANSITION の targets に渡されていること\n"
            "  - TRANSITION が vector<int> を対象として DECK_BOTTOM へ移動すること"
        )

    def test_hand_net_change_correct_with_target_selection(self) -> None:
        """N枚ドロー・N枚返却後の手札純増がゼロであること（戻した枚数 = 引いた枚数）。

        バグ修正前: 返却されないため手札が純増し続ける。
        バグ修正後: draw N → select N to return → 手札純増ゼロ。
        """
        game, db = _setup(self.CARD_ID, self.COST, seed=30)
        s = game.state

        play_cmd = _cmd(game, db, "PLAY")
        if play_cmd is None:
            pytest.skip("PLAY コマンドなし")

        hand_before = len(s.players[0].hand)
        game.resolve_command(play_cmd)
        hand_after_play = len(s.players[0].hand)  # プレイで -1

        resolve_cmd = _cmd(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドなし")
        game.resolve_command(resolve_cmd)

        # SELECT_NUMBER: 1枚ドロー
        sel_num = _cmd(game, db, "SELECT_NUMBER")
        drawn_count = 0
        if sel_num is not None:
            choose_1 = next(
                (c for c in _all_cmds(game, db)
                 if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
                 and getattr(c, "target_instance", -1) == 1),
                sel_num,
            )
            drawn_count = getattr(choose_1, "target_instance", 0)
            game.resolve_command(choose_1)

        # SELECT_TARGET: 引いた枚数だけ手札から選ぶ
        returned = 0
        for _ in range(drawn_count + 5):
            legal = _all_cmds(game, db)
            tgt = next(
                (c for c in legal
                 if "SELECT_TARGET" in str(getattr(c, "type", "")).upper()),
                None,
            )
            if tgt is None:
                break
            game.resolve_command(tgt)
            returned += 1
            if not s.waiting_for_user_input:
                break

        hand_after_effect = len(s.players[0].hand)
        delta = hand_after_effect - hand_after_play  # ドロー - 返却

        # drawn_count 分だけドローし、drawn_count 分返却 → 純増ゼロ
        # ただし SELECT_TARGET 実装前のフォールバック: delta > drawn_count はバグ
        assert delta <= drawn_count, (
            f"手札純増 {delta} が引いた枚数 {drawn_count} を超えています。\n"
            f"  hand_after_play={hand_after_play}, hand_after_effect={hand_after_effect}\n"
            "返却コマンドが機能していない可能性があります。"
        )
        assert returned == drawn_count, (
            f"返却回数 {returned} が引いた枚数 {drawn_count} と一致しません。\n"
            "SELECT_TARGET が正しい回数生成されていません。"
        )

    def test_select_target_count_equals_hand_size(self) -> None:
        """SELECT_TARGET の候補数が発動プレイヤーの手札枚数と一致すること。

        バグ修正前:
          handle_select が filter.owner を無視して両プレイヤーのゾーンを走査し、
          相手の手札カードも valid_targets に含めてしまう。
          その結果 SELECT_TARGET コマンドが手札枚数を超えて生成される。
          例: 自分手札 4 枚 + 相手手札 5 枚 = 9 件 が生成される。

        バグ修正後:
          filter に owner="SELF" を設定し、発動プレイヤーの手札のみを走査する。
          SELECT_TARGET コマンド数 == 発動プレイヤーの手札枚数。

        再発防止:
          - QUERY(SELECT_TARGET) の CommandDef に target_group=PLAYER_SELF を設定すること
          - generate_primitive_instructions で owner="SELF" を filter に付加すること
          - handle_select は filter.owner の有無に応じて走査対象を限定すること
        """
        game, db = _setup(self.CARD_ID, self.COST, seed=99)
        s = game.state

        play_cmd = _cmd(game, db, "PLAY")
        if play_cmd is None:
            pytest.skip("PLAY コマンドなし")
        game.resolve_command(play_cmd)

        resolve_cmd = _cmd(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドなし")
        game.resolve_command(resolve_cmd)

        # SELECT_NUMBER で 1 枚選択
        sel_num = _cmd(game, db, "SELECT_NUMBER")
        if sel_num is None:
            pytest.skip("SELECT_NUMBER コマンドなし")
        choose_1 = next(
            (c for c in _all_cmds(game, db)
             if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()
             and getattr(c, "target_instance", -1) == 1),
            sel_num,
        )
        game.resolve_command(choose_1)

        # SELECT_TARGET 生成時に候補数 == 発動プレイヤーの手札枚数であること
        hand_size_p0 = len(s.players[0].hand)
        legal = _all_cmds(game, db)
        sel_targets = [
            c for c in legal
            if "SELECT_TARGET" in str(getattr(c, "type", "")).upper()
        ]

        assert len(sel_targets) == hand_size_p0, (
            f"SELECT_TARGET 候補数 {len(sel_targets)} が"
            f"発動プレイヤーの手札枚数 {hand_size_p0} と一致しません。\n"
            f"  相手の手札カードが候補に混入している可能性があります。\n"
            "  再発防止:\n"
            "  - filter.owner を SELF に設定し、発動プレイヤーの手札のみ走査すること\n"
            "  - handle_select は owner 未設定時は active_player のみを走査すること"
        )
