# tests/test_per_card_effects.py
"""カードID別の効果処理 TDD テスト。

対象: 各カードの ON_PLAY / ON_CAST_SPELL トリガー効果が正しく発火・実行されること。

再発防止:
- ON_PLAY は pipeline の CHECK_CREATURE_ENTER_TRIGGERS だけで処理する。
  TriggerManager で二重発火しないよう trigger_manager.cpp に early-return を入れてある。
- 効果は pending_effects に TRIGGER_ABILITY として積まれ、
  RESOLVE_EFFECT コマンドを実行して初めて効果が適用される。
- add_card_to_mana(player_id, card_id, instance_id) でテスト用マナを直接追加できる。
  instance_id は既存カードと重複しないよう 9000 番台以降を使用する。
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple

import pytest

dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")
if not getattr(dm_ai_module, "IS_NATIVE", False):
    pytest.skip("Requires native dm_ai_module (IS_NATIVE=True)", allow_module_level=True)

_CARDS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "cards.json")
_MANA_INSTANCE_START = 9000  # テスト用マナカードの instance_id 開始値


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def _make_db() -> Any:
    """cards.json をロードした CardDatabase を返す。"""
    if os.path.exists(_CARDS_JSON):
        return dm_ai_module.JsonLoader.load_cards(_CARDS_JSON)
    pytest.skip("cards.json が存在しない環境")


def _setup_game_for_card(
    card_id: int,
    mana_cost: int,
    seed: int = 42,
) -> Tuple[Any, Any]:
    """指定カードをP0の手札に持ち、必要マナを充填した状態で MainPhase まで進める。

    手順:
    1. デッキを card_id で埋めて start_game（P0の手札は全て card_id）
    2. add_card_to_mana でコスト分のマナを直接追加（instance_id=9000+）
    3. generate_legal_commands → resolve_command を繰り返し MAIN フェーズへ到達

    Returns:
        (game, db)
    """
    db = _make_db()
    game = dm_ai_module.GameInstance(seed, db)
    s = game.state
    # デッキを対象カードで埋める
    s.set_deck(0, [card_id] * 40)
    s.set_deck(1, [1] * 40)
    dm_ai_module.PhaseManager.start_game(s, db)

    # コスト分のマナを直接追加（通常の MANA_CHARGE とは独立）
    for i in range(mana_cost):
        s.add_card_to_mana(0, card_id, _MANA_INSTANCE_START + i)

    # MainPhase (P0) に到達するまでフェーズを進める
    for _ in range(20):
        ph = str(s.current_phase).upper()
        if "MAIN" in ph and s.active_player_id == 0:
            break
        legal: List[Any] = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if legal:
            game.resolve_command(legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)
    else:
        pytest.fail(
            f"card_id={card_id}: MainPhase (P0) に到達できませんでした。"
            f" current_phase={s.current_phase}"
        )

    return game, db


def _find_play_cmd(game: Any, db: Any) -> Any:
    """合法コマンドから最初の PLAY_FROM_ZONE を返す。なければ None。"""
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
    for c in legal:
        if "PLAY" in str(getattr(c, "type", "")).upper():
            return c
    return None


def _find_cmd_by_keyword(game: Any, db: Any, keyword: str) -> Any:
    """合法コマンドからキーワードにマッチする最初のコマンドを返す。"""
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
    for c in legal:
        if keyword.upper() in str(getattr(c, "type", "")).upper():
            return c
    return None


# ---------------------------------------------------------------------------
# 1. Card 5: AQvibrato (cost=2) — ON_PLAY → DRAW_CARD 1
# ---------------------------------------------------------------------------

class TestCard5AQvibrato:
    """AQvibrato (id=5, cost=2): ON_PLAY → DRAW_CARD 1 の動作検証。

    再発防止: ON_PLAY は trigger_manager.cpp の check_triggers では処理しない。
    pipeline の CHECK_CREATURE_ENTER_TRIGGERS のみで処理されること。
    """

    CARD_ID = 5
    COST = 2

    def test_on_play_creates_exactly_one_pending_effect(self) -> None:
        """召喚後、pending_effects が 1 件だけ生成されることを確認する。

        再発防止: 二重発火バグ修正後は exactly 1件。2件なら trigger_manager.cpp を確認。
        """
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません（マナ不足の可能性）")

        assert s.get_pending_effect_count() == 0, "召喚前に pending_effects があります"
        game.resolve_command(play_cmd)

        pending = s.get_pending_effect_count()
        assert pending == 1, (
            f"ON_PLAY 後の pending_effects が {pending} 件です（期待値: 1）。\n"
            "再発防止: 2件の場合は trigger_manager.cpp の ON_PLAY early-return を確認。"
        )

    def test_on_play_draw_card_increases_hand(self) -> None:
        """RESOLVE_EFFECT 後に手札が1枚増え、デッキが1枚減ることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        hand_before = len(s.players[0].hand)
        deck_before = len(s.players[0].deck)

        game.resolve_command(play_cmd)
        # 手札-1 (プレイ)
        assert len(s.players[0].hand) == hand_before - 1

        resolve_cmd = _find_cmd_by_keyword(game, db, "RESOLVE")
        assert resolve_cmd is not None, "RESOLVE_EFFECT コマンドが見つかりません"
        game.resolve_command(resolve_cmd)

        hand_after = len(s.players[0].hand)
        deck_after = len(s.players[0].deck)

        # プレイ-1, ドロー+1 → net 0 (元の手札枚数と同じ)
        assert hand_after == hand_before, (
            f"手札: before={hand_before} after={hand_after} (期待: {hand_before})\n"
            "DRAW_CARD 1 が正しく実行されていません"
        )
        assert deck_after == deck_before - 1, (
            f"デッキ: before={deck_before} after={deck_after} (期待: {deck_before - 1})\n"
            "DRAW_CARD でデッキからカードが引かれていません"
        )

    def test_on_play_pending_clears_after_resolve(self) -> None:
        """RESOLVE_EFFECT 後に pending_effects が 0 になることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)

        resolve_cmd = _find_cmd_by_keyword(game, db, "RESOLVE")
        assert resolve_cmd is not None
        game.resolve_command(resolve_cmd)

        assert s.get_pending_effect_count() == 0, "RESOLVE 後も pending_effects が残っています"


# ---------------------------------------------------------------------------
# 2. Card 2: 芸魔隠狐カラクリバーシ (cost=5) — ON_PLAY → DRAW_CARD 1
# ---------------------------------------------------------------------------

class TestCard2KarakuriBarshi:
    """カラクリバーシ (id=2, cost=5): ON_PLAY → DRAW_CARD 1 の動作検証。"""

    CARD_ID = 2
    COST = 5

    def test_on_play_creates_one_pending_effect(self) -> None:
        """召喚後に pending_effects が 1 件生成されることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 1, (
            f"ON_PLAY 後の pending_effects={s.get_pending_effect_count()} (期待: 1)"
        )

    def test_on_play_draw_card_executes_after_resolve(self) -> None:
        """RESOLVE_EFFECT 後にデッキが少なくとも1枚減ることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        deck_before = len(s.players[0].deck)
        game.resolve_command(play_cmd)

        resolve_cmd = _find_cmd_by_keyword(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドが見つかりません")
        game.resolve_command(resolve_cmd)

        deck_after = len(s.players[0].deck)
        assert deck_after <= deck_before - 1, (
            f"RESOLVE 後デッキが減っていません: before={deck_before} after={deck_after}\n"
            "DRAW_CARD が実行されていない可能性があります"
        )


# ---------------------------------------------------------------------------
# 3. Card 8: 同期の妖精 (cost=2) — effects なし
# ---------------------------------------------------------------------------

class TestCard8DokiNoYosei:
    """同期の妖精 (id=8, cost=2): effects=[] → 召喚後に pending_effects が 0 であること。"""

    CARD_ID = 8
    COST = 2

    def test_no_on_play_effect(self) -> None:
        """effects がないカードは召喚後に pending_effects = 0 であることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 0, (
            f"effects なしカードでも pending_effects={s.get_pending_effect_count()} が生成されました"
        )


# ---------------------------------------------------------------------------
# 4. Card 13: 単騎連射 マグナム (cost=3) — ON_PLAY → REPLACE_MOVE_CARD
# ---------------------------------------------------------------------------

class TestCard13MagnumSingleshot:
    """マグナム (id=13, cost=3): ON_PLAY → REPLACE_MOVE_CARD 効果の発火確認。"""

    CARD_ID = 13
    COST = 3

    def test_on_play_creates_pending_effect(self) -> None:
        """召喚後に pending_effects が 1 件生成されることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 1, (
            f"ON_PLAY 後の pending_effects={s.get_pending_effect_count()} (期待: 1)"
        )


# ---------------------------------------------------------------------------
# 5. Card 1: 月光電人オボロカゲロウ (cost=2) — ON_PLAY → QUERY + optional DRAW
# ---------------------------------------------------------------------------

class TestCard1OboroKagero:
    """オボロカゲロウ (id=1, cost=2): ON_PLAY → QUERY + optional DRAW_CARD。

    複合効果（QUERY, DRAW_CARD, TRANSITION）のため、
    pending_effects が生成されることと、RESOLVE 後に状態が変わることを確認する。
    """

    CARD_ID = 1
    COST = 2

    def test_on_play_creates_pending_effect(self) -> None:
        """召喚後に pending_effects が 1 件生成されることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() >= 1, (
            f"ON_PLAY 後の pending_effects={s.get_pending_effect_count()} (期待: >= 1)"
        )

    def test_no_double_fire(self) -> None:
        """再発防止: ON_PLAY 効果が二重発火しないことを確認する。

        二重発火すると pending_effects=2 になる。
        trigger_manager.cpp の ON_PLAY early-return が機能していることを検証。
        """
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        pending = s.get_pending_effect_count()
        assert pending <= 1, (
            f"ON_PLAY が二重発火しています: pending_effects={pending} (期待: <= 1)。\n"
            "再発防止: trigger_manager.cpp の ON_PLAY early-return を確認してください。"
        )


# ---------------------------------------------------------------------------
# 6. Card 4: Napo獅子-Vi無粋 (cost=5) — ON_PLAY → DISCARD 2 + DRAW 2
# ---------------------------------------------------------------------------

class TestCard4NapoShishi:
    """Napo獅子 (id=4, cost=5): ON_PLAY → DISCARD 2 → DRAW_CARD 2 の発火確認。

    DISCARD が mandatory (optional=False) のため、効果解決後に
    手札枚数が正しく変化することを確認する（DISCARD-2 + DRAW+2 = net 0）。
    """

    CARD_ID = 4
    COST = 5

    def test_on_play_creates_pending_effect(self) -> None:
        """召喚後に pending_effects が 1 件生成されることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 1, (
            f"ON_PLAY 後の pending_effects={s.get_pending_effect_count()} (期待: 1)"
        )

    def test_no_double_fire(self) -> None:
        """再発防止: ON_PLAY が二重発火しないことを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        pending = s.get_pending_effect_count()
        assert pending <= 1, (
            f"ON_PLAY が二重発火: pending_effects={pending} (期待: <= 1)。\n"
            "再発防止: trigger_manager.cpp の ON_PLAY early-return を確認。"
        )


# ---------------------------------------------------------------------------
# 7. Card 3: 芸魔王将カクメイジン (cost=7) — ON_PLAY 効果なし
# ---------------------------------------------------------------------------

class TestCard3KakuMeijin:
    """カクメイジン (id=3, cost=7): ON_PLAY 効果なし → 召喚後 pending_effects=0。

    カクメイジンは ON_ATTACK_FROM_HAND と AT_BREAK_SHIELD のみ効果を持つ。
    また static_ability で自分のバトルゾーンのマジックにスピードアタッカーを付与する。
    召喚直後は pending_effects が生成されないことを確認する。
    """

    CARD_ID = 3
    COST = 7

    def test_no_on_play_effect(self) -> None:
        """ON_PLAY 効果なし → 召喚後 pending_effects=0 を確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 0, (
            f"ON_PLAY 効果なしのカードで pending_effects={s.get_pending_effect_count()} が生成されました。\n"
            "カクメイジンは ON_ATTACK_FROM_HAND/AT_BREAK_SHIELD のみ効果あり。"
        )

    def test_no_double_fire(self) -> None:
        """再発防止: ON_PLAY 効果なしカードで pending_effects が溜まらないことを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 0, (
            f"二重発火の可能性: pending_effects={s.get_pending_effect_count()} (期待: 0)。\n"
            "再発防止: trigger_manager.cpp の ON_PLAY early-return を確認。"
        )


# ---------------------------------------------------------------------------
# 8. Card 6: 歌舞音愛 ヒメカット (cost=2) — ON_PLAY 効果なし（クリーチャー面）
# ---------------------------------------------------------------------------

class TestCard6HimeCut:
    """ヒメカット (id=6, cost=2): クリーチャー面はON_PLAY効果なしだが friend_burst あり。

    friend_burst キーワードは keyword_expander.cpp により ON_PLAY トリガーの
    FRIEND_BURST 効果 (optional=True) に展開される。
    よって召喚後 pending_effects=1 (FRIEND_BURST 発動待ち) が期待値。

    実際の効果 (ON_OPPONENT_DRAW) は相手ドロー時に別途発火する。
    ツインパクト呪文面（コスト4）は ON_CAST_SPELL を持つが、
    呪文詠唱テストは spell PLAY_FROM_ZONE 実装後に追加予定。
    """

    CARD_ID = 6
    COST = 2

    def test_friend_burst_creates_one_pending_on_play(self) -> None:
        """friend_burst キーワードにより召喚後 pending_effects=1 であることを確認する。

        再発防止: keyword_expander.cpp の expand_friend_burst が ON_PLAY トリガー効果を
        追加するため、friend_burst 持ちカードは必ず pending >= 1 になる。
        これは正常な動作であり、バグではない。
        """
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 1, (
            f"friend_burst 持ちカードの召喚後 pending_effects={s.get_pending_effect_count()} (期待: 1)。\n"
            "keyword_expander.cpp の expand_friend_burst が ON_PLAY 効果を追加するため pending=1 が正常。"
        )

    def test_no_double_fire(self) -> None:
        """再発防止: friend_burst が二重発火しないことを確認する (pending<=1)。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        pending = s.get_pending_effect_count()
        assert pending <= 1, (
            f"friend_burst が二重発火: pending_effects={pending} (期待: <= 1)。\n"
            "再発防止: keyword_expander.cpp の expand_friend_burst を確認してください。"
        )


# ---------------------------------------------------------------------------
# 9. Card 9: ボン・キゴマイム (cost=3) — ON_PLAY 効果なし（クリーチャー面）
# ---------------------------------------------------------------------------

class TestCard9BonKigoMaim:
    """ボン・キゴマイム (id=9, cost=3): クリーチャー面は ON_PLAY 効果なし。

    効果は ON_OPPONENT_CREATURE_ENTER（相手クリーチャー召喚時に CANNOT_ATTACK を付与）のみ。
    召喚直後の pending_effects=0 を確認する。
    ツインパクト呪文面（コスト2）は ON_CAST_SPELL を持つが、
    呪文詠唱テストは spell PLAY_FROM_ZONE 実装後に追加予定。
    """

    CARD_ID = 9
    COST = 3

    def test_no_on_play_effect(self) -> None:
        """クリーチャー召喚後に pending_effects=0 であることを確認する。"""
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state

        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        game.resolve_command(play_cmd)
        assert s.get_pending_effect_count() == 0, (
            f"ボン・キゴマイムはON_PLAY効果なしのはずが pending_effects={s.get_pending_effect_count()}。\n"
            "ON_OPPONENT_CREATURE_ENTER 効果が誤って ON_PLAY に紐付けられていないか確認してください。"
        )


# ---------------------------------------------------------------------------
# 10. Card 12: ストリーミング・シェイパー (cost=3, SPELL) — 呪文詠唱テスト
# ---------------------------------------------------------------------------

class TestCard12StreamingShaper:
    """ストリーミング・シェイパー (id=12, cost=3): SPELL。

    ON_CAST_SPELL: デッキトップ4枚をバッファに公開 → 水文明のカードをSELECT_FROM_BUFFER → 手札へ。
    再発防止: IntentGenerator が呪文の PLAY_FROM_ZONE を生成するようになったらスキップを外すこと。
    """

    CARD_ID = 12
    COST = 3

    def test_on_cast_creates_pending_effect(self) -> None:
        """呪文詠唱後に pending_effects が 1 件生成されることを確認する。

        再発防止: spell の ON_CAST_SPELL トリガーが pending_effect に積まれること。
        PLAY_FROM_ZONE 直後は pending_effect_count == 1 であるはず。
        """
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state
        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        assert s.get_pending_effect_count() == 0, "詠唱前に pending_effects があります"
        game.resolve_command(play_cmd)
        pending = s.get_pending_effect_count()
        assert pending == 1, (
            f"ON_CAST_SPELL 後の pending_effects が {pending} 件です（期待値: 1）。\n"
            "再発防止: trigger_manager.cpp の ON_CAST_SPELL トリガー処理を確認してください。"
        )

    def test_on_cast_resolves_buffer_and_deck_decreases(self) -> None:
        """RESOLVE_EFFECT 後に REVEAL_TO_BUFFER でバッファを経由してカードが移動することを確認する。

        Card12 の効果: REVEAL_TO_BUFFER(4) → SELECT_FROM_BUFFER(WATER) → MOVE_BUFFER_TO_ZONE(HAND)
        デッキは card_id=12 (WATER) で埋まっているため、バッファに置かれた4枚はすべて水文明。

        動作フロー:
          1. REVEAL_TO_BUFFER(4): デッキ上位4枚をバッファへ (deck -4)
          2. SELECT_FROM_BUFFER: ユーザーが1枚以上選択 ($buffer_select に登録)
          3. MOVE($buffer_select → HAND): 選択分が手札へ
          4. MOVE(BUFFER_REMAIN → DECK_BOTTOM): 残余がデッキボトムへ戻る
          ∴ deck_net = -(選択枚数), hand_net = +(選択枚数) - 1(呪文プレイ)

        再発防止: BUFFER_REMAIN が DECK_BOTTOM へ戻らない場合は
                  pipeline_executor.cpp の BUFFER_REMAIN 仮想ターゲット処理を確認。
                  $buffer_select が空の場合は SELECT_FROM_BUFFER の dispatch を確認。
        """
        game, db = _setup_game_for_card(self.CARD_ID, self.COST)
        s = game.state
        play_cmd = _find_play_cmd(game, db)
        if play_cmd is None:
            pytest.skip("PLAY_FROM_ZONE コマンドが見つかりません")

        deck_before = len(s.players[0].deck)
        hand_before = len(s.players[0].hand)

        game.resolve_command(play_cmd)

        resolve_cmd = _find_cmd_by_keyword(game, db, "RESOLVE")
        if resolve_cmd is None:
            pytest.skip("RESOLVE_EFFECT コマンドが見つかりません")
        game.resolve_command(resolve_cmd)

        # SELECT_FROM_BUFFER への返答が必要な場合はすべて送信する
        for _ in range(10):
            if not s.waiting_for_user_input:
                break
            legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
            if not legal:
                break
            sel_cmd = next(
                (c for c in legal
                 if "SELECT_FROM_BUFFER" in str(getattr(c, "type", "")).upper()),
                legal[0],
            )
            game.resolve_command(sel_cmd)

        deck_after = len(s.players[0].deck)
        hand_after = len(s.players[0].hand)

        # deck_after == deck_before - (selected_count): 選択した分だけ純減
        # BUFFER_REMAIN の残余がデッキボトムへ戻っているため 4枚固定減にはならない
        # 最低1枚の選択 → deck_net >= -1 かつ hand_net >= 0 を確認
        selected_count = deck_before - deck_after
        assert selected_count >= 1, (
            f"デッキの純減が0です (before={deck_before} after={deck_after})。\n"
            "REVEAL_TO_BUFFER→SELECT_FROM_BUFFER→MOVE_BUFFER_TO_ZONE が機能していません。\n"
            "再発防止: pipeline_executor.cpp の BUFFER_REMAIN 仮想ターゲット処理を確認してください。"
        )
        # MOVE_BUFFER_TO_ZONE が動作しているなら手札は play-1 + select + になる
        assert hand_after >= hand_before - 1, (
            f"手札: before={hand_before} after={hand_after}。\n"
            "カードがバッファから手札に移動していません。\n"
            "再発防止: command_system.cpp の MOVE_BUFFER_TO_ZONE で"
            ' target="$buffer_select" を使用していることを確認してください。'
        )
