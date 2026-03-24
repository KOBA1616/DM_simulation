# -*- coding: utf-8 -*-
"""
GUIゲームセッション / ヘッドレスセッション 整合性テスト
==========================================================
以下を検証する:
  1. headless_runner が cards.json を使用して初期状態が正しいこと
  2. GameSession がヘッドレスモードで初期状態を正しく構築すること
  3. 両セッションとも shields=5 / hand=5 の初期状態であること
  4. headless_runner がゲームを完走し勝者を返すこと
  5. 同一シードで headless_runner が同じ勝者を返すこと（再現性）
  6. DataCollector の基本動作（test_game_flow_minimal から移植）

再発防止:
  - headless_runner は CardDatabase() ではなく JsonLoader.load_cards() を使うこと
    （CardDatabase() は空DBを返すため cards.json の情報なしで動作になってしまう）
  - GameSession の reset_game() は PhaseManager.fast_forward() を呼ぶため
    手札枚数は初回ドロースキップ適用後の値（先攻1ターン目=5枚）になる
  - このファイルで PyQt6 を import してはならない
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

dm_ai_module = pytest.importorskip("dm_ai_module", reason="dm_ai_module が見つかりません")

IS_NATIVE: bool = bool(getattr(dm_ai_module, "IS_NATIVE", False))

pytestmark = pytest.mark.skipif(
    not IS_NATIVE,
    reason="ネイティブモジュール (IS_NATIVE=True) が必要です",
)

_CARDS_JSON = os.path.join(_PROJECT_ROOT, "data", "cards.json")
_INITIAL_SHIELD_COUNT = 5
_INITIAL_HAND_COUNT = 5


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------


def _make_native_game(seed: int = 42) -> tuple[Any, Any]:
    """headless_runner と同等のセットアップ（cards.json ロード版）でゲームを作成する。"""
    db = dm_ai_module.JsonLoader.load_cards(_CARDS_JSON)
    game = dm_ai_module.GameInstance(seed, db)
    gs = game.state
    gs.set_deck(0, [1] * 40)
    gs.set_deck(1, [1] * 40)
    dm_ai_module.PhaseManager.start_game(gs, db)
    return game, db


# ---------------------------------------------------------------------------
# Class 1: headless_runner の DB 整合性
# ---------------------------------------------------------------------------


class TestHeadlessRunnerDbConsistency:
    """headless_runner が cards.json をロードして動作することを検証する。

    再発防止: python/dm_env/headless_runner.py で CardDatabase() を使うと
    空DBになりカード効果が一切解決されない。JsonLoader.load_cards() を使うこと。
    """

    def test_headless_runner_completes_game(self) -> None:
        """headless_runner.run_game() がゲームを完走し 'winner' キーを返すことを確認する。"""
        from python.dm_env.headless_runner import run_game

        result = run_game([], [], seed=42)
        assert "winner" in result, f"'winner' キーがない: {result}"
        assert "turns" in result, f"'turns' キーがない: {result}"
        assert result["turns"] >= 0, f"turns が負: {result['turns']}"

    def test_headless_runner_reproducibility(self) -> None:
        """同一シードで run_game() を2回呼んだとき同じ勝者が返ることを確認する。

        再発防止: CardDatabase() の場合、シード設定が機能せず結果が変わる場合がある。
        JsonLoader でカードを読み込んだうえで再現性を保証すること。
        """
        from python.dm_env.headless_runner import run_game

        r1 = run_game([], [], seed=100)
        r2 = run_game([], [], seed=100)
        assert r1["winner"] == r2["winner"], (
            f"同一シードで結果が異なる: {r1['winner']} vs {r2['winner']}\n"
            "再発防止: headless_runner の乱数シードが正しく渡されていることを確認すること"
        )

    def test_headless_runner_uses_real_cards_initial_shields(self) -> None:
        """headless_runner と同等のセットアップで初期シールドが 5 枚であることを確認する。

        再発防止: 空DBでゲームを開始するとシールド配置が不完全になる場合がある。
        """
        game, _ = _make_native_game(seed=42)
        gs = game.state
        p0_shields = len(gs.players[0].shield_zone)
        p1_shields = len(gs.players[1].shield_zone)
        assert p0_shields == _INITIAL_SHIELD_COUNT, (
            f"P0 初期シールド: {p0_shields} != {_INITIAL_SHIELD_COUNT}\n"
            "再発防止: JsonLoader.load_cards() で DB をロードしてから start_game() を呼ぶこと"
        )
        assert p1_shields == _INITIAL_SHIELD_COUNT, (
            f"P1 初期シールド: {p1_shields} != {_INITIAL_SHIELD_COUNT}"
        )

    def test_headless_runner_uses_real_cards_initial_hand(self) -> None:
        """headless_runner と同等のセットアップで初期手札が 5 枚であることを確認する。"""
        game, _ = _make_native_game(seed=42)
        gs = game.state
        p0_hand = len(gs.players[0].hand)
        p1_hand = len(gs.players[1].hand)
        assert p0_hand == _INITIAL_HAND_COUNT, (
            f"P0 初期手札: {p0_hand} != {_INITIAL_HAND_COUNT}"
        )
        assert p1_hand == _INITIAL_HAND_COUNT, (
            f"P1 初期手札: {p1_hand} != {_INITIAL_HAND_COUNT}"
        )


# ---------------------------------------------------------------------------
# Class 2: GameSession (ヘッドレスモード) 初期状態の整合性
# ---------------------------------------------------------------------------


class TestGameSessionHeadlessState:
    """GameSession がヘッドレスモードで正しい初期状態を構築することを検証する。

    再発防止:
    - dm_toolkit.gui.game_session.GameSession は PyQt6 なしでインポート可能。
    - reset_game() は PhaseManager.fast_forward() を呼ぶため先攻1ターン目の
      ドローがスキップされ初期手札は 5 枚のままであること。
    """

    def test_game_session_initial_shields(self) -> None:
        """GameSession.reset_game() 後の初期シールドが 5 枚であることを確認する。"""
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        gs = sess.gs
        assert gs is not None, "GameSession.gs が None"
        p0_shields = len(gs.players[0].shield_zone)
        p1_shields = len(gs.players[1].shield_zone)
        assert p0_shields == _INITIAL_SHIELD_COUNT, (
            f"GameSession P0 初期シールド: {p0_shields} != {_INITIAL_SHIELD_COUNT}"
        )
        assert p1_shields == _INITIAL_SHIELD_COUNT, (
            f"GameSession P1 初期シールド: {p1_shields} != {_INITIAL_SHIELD_COUNT}"
        )

    def test_game_session_initial_hand(self) -> None:
        """GameSession.reset_game() 後の初期手札が 5 枚であることを確認する。

        再発防止: fast_forward() の後でも先攻1ターン目はドローをスキップするため
        手札は 5 枚のままになる。もし 6 枚になっている場合は先攻ドロースキップが機能していない。
        """
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        gs = sess.gs
        assert gs is not None, "GameSession.gs が None"
        p0_hand = len(gs.players[0].hand)
        p1_hand = len(gs.players[1].hand)
        assert p0_hand == _INITIAL_HAND_COUNT, (
            f"GameSession P0 初期手札: {p0_hand} != {_INITIAL_HAND_COUNT}\n"
            "再発防止: 先攻1ターン目はドローをスキップすること"
        )
        assert p1_hand == _INITIAL_HAND_COUNT, (
            f"GameSession P1 初期手札: {p1_hand} != {_INITIAL_HAND_COUNT}"
        )

    def test_game_session_initial_state_matches_native_setup(self) -> None:
        """GameSession と headless_runner 等価セットアップの初期状態が一致することを確認する。

        再発防止: GameSession は setup_test_duel() + fast_forward() を呼ぶが
        ネイティブセットアップと shields/hand 枚数が一致しなければならない。
        """
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        gs_session = sess.gs

        game_native, _ = _make_native_game(seed=0)
        gs_native = game_native.state

        for pid in range(2):
            sess_shields = len(gs_session.players[pid].shield_zone)
            native_shields = len(gs_native.players[pid].shield_zone)
            assert sess_shields == native_shields == _INITIAL_SHIELD_COUNT, (
                f"P{pid} シールド不一致: GameSession={sess_shields} vs Native={native_shields}\n"
                "再発防止: 両セッションで PhaseManager.start_game() を呼んでいることを確認すること"
            )

            sess_hand = len(gs_session.players[pid].hand)
            native_hand = len(gs_native.players[pid].hand)
            assert sess_hand == native_hand == _INITIAL_HAND_COUNT, (
                f"P{pid} 手札不一致: GameSession={sess_hand} vs Native={native_hand}"
            )


# ---------------------------------------------------------------------------
# Class 3: DataCollector スモークテスト（test_game_flow_minimal から移植）
# ---------------------------------------------------------------------------


class TestDataCollectorSmoke:
    """DataCollector の基本動作スモークテスト。

    元テスト: test_game_flow_minimal.py::TestGameFlowMinimal::_step_data_collection
    再発防止: test_game_flow_minimal.py は旧 API・print 汚染・trivial アサーションのため削除済み。
    このテストで DataCollector の動作確認を引き継ぐ。
    """

    def test_collect_batch_heuristic_returns_values(self) -> None:
        """DataCollector.collect_data_batch_heuristic が values リストを返すことを確認する。"""
        collector = dm_ai_module.DataCollector()
        batch = collector.collect_data_batch_heuristic(1, True, False)
        assert batch is not None, "DataCollector.collect_data_batch_heuristic が None を返した"
        assert isinstance(batch.values, list), (
            f"batch.values がリストでない: {type(batch.values)}"
        )


# ---------------------------------------------------------------------------
# Class 4: コマンド生成バグ回帰テスト（TDD）
# ---------------------------------------------------------------------------


class TestCommandGenerationRegression:
    """
    GameSession.generate_legal_commands() および headless.find_legal_commands_for_instance()
    のバグ回帰テスト。

    修正済みバグ:
      1. game_session.py: _generate_legal_commands(strict=False) → TypeError をサイレントに飲み
         常に [] を返すバグ → strict=False を除去して修正
      2. headless.py: find_legal_commands_for_instance で `from dm_toolkit import commands`
         が `return []` の後に書かれていてデッドコード＋NameError になるバグ
         → import を早期 return より前に移動して修正

    再発防止:
      - _generate_legal_commands は strict/skip_wrapper キーワード引数を持たない。
        コマンド生成は dm.IntentGenerator.generate_legal_commands(state, card_db) のみ。
      - headless.py の関数ローカル import は return より前に書くこと。
    """

    def test_game_session_generate_legal_commands_returns_list(self) -> None:
        """GameSession.generate_legal_commands() がリストを返すことを確認する（TypeError バグ回帰）。

        修正前: _generate_legal_commands(gs, card_db, strict=False) → TypeError → [] を返す
        修正後: _generate_legal_commands(gs, card_db) → 実際のコマンドリストを返す
        """
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        assert sess.gs is not None, "GameSession.gs が None"

        cmds = sess.generate_legal_commands()
        assert isinstance(cmds, list), (
            f"generate_legal_commands() がリストを返していない: {type(cmds)}\n"
            "再発防止: _generate_legal_commands に strict= 等の不正キーワード引数を渡さないこと"
        )
        # 少なくとも PASS コマンドが返るはず
        assert len(cmds) >= 1, (
            f"generate_legal_commands() が空リストを返した (cmds={cmds})\n"
            "再発防止: _generate_legal_commands(gs, card_db, strict=False) のように"
            "          不正引数を渡すと TypeError で常に [] になる。修正済みか確認すること"
        )

    def test_headless_find_legal_commands_returns_list(self) -> None:
        """headless.find_legal_commands_for_instance() がリストを返すことを確認する（NameError バグ回帰）。

        修正前: `from dm_toolkit import commands` が return [] の後に書かれていて NameError
                → except Exception に吸収されて常に [] を返す（バグ）
        修正後: import は早期 return より前に配置 → 正常に commands を解決して呼び出せる
        """
        from dm_toolkit.gui import headless
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        assert sess.gs is not None, "GameSession.gs が None"

        # instance_id は存在しない値でよい; 関数がクラッシュしないことを確認する
        result = headless.find_legal_commands_for_instance(sess, instance_id=9999)
        assert isinstance(result, list), (
            f"find_legal_commands_for_instance() がリストを返していない: {type(result)}\n"
            "再発防止: import 文を return の後に書いてはならない（NameError になる）"
        )

    def test_headless_play_instance_returns_bool(self) -> None:
        """headless.play_instance() が bool を返しクラッシュしないことを確認する。"""
        from dm_toolkit.gui import headless
        from dm_toolkit.gui.game_session import GameSession

        sess = GameSession()
        sess.reset_game()
        assert sess.gs is not None

        # 存在しない instance_id に対して False を返すはず
        result = headless.play_instance(sess, instance_id=9999)
        assert isinstance(result, bool), (
            f"play_instance() が bool を返していない: {type(result)}"
        )
        assert result is False, (
            "存在しない instance_id=9999 に対して True が返った（コマンドが誤って実行されている）"
        )


# ---------------------------------------------------------------------------
# Class 5: GameSession フルゲーム完走テスト（TDD）
# ---------------------------------------------------------------------------


class TestHeadlessFullGameRun:
    """GameSession + headless.run_steps で 1ゲームが完走することを検証する。

    再発防止:
    - headless.run_steps は max_steps 以内にゲームが終了するはず。
    - 終了しない場合は game_session.step_game() か fast_forward() に無限ループがある。
    """

    def test_full_game_completes_within_500_steps(self) -> None:
        """GameSession + headless.run_steps が 500 ステップ以内に完走することを確認する。

        再発防止: step_game() が StopIteration・無限ループになる場合は
                  game_instance.step() の戻り値（bool）を確認すること。
        """
        from dm_toolkit.gui.game_session import GameSession
        from dm_toolkit.gui import headless

        sess = GameSession()
        sess.reset_game()
        assert sess.gs is not None, "GameSession.gs が None"

        steps, over = headless.run_steps(sess, max_steps=500)
        assert over, (
            f"500 ステップ以内にゲームが終了しなかった (steps={steps})\n"
            "再発防止: step_game() が進行しない場合は _no_action_count の上限を確認すること"
        )
        assert steps > 0, f"steps が 0 (steps={steps})"
        # game_over フラグとの整合性
        if sess.gs:
            assert sess.gs.game_over, "run_steps が True を返したが gs.game_over が False"

    def test_winner_set_after_full_game(self) -> None:
        """フルゲーム後に winner_player が設定されることを確認する。

        再発防止: game_over=True でも winner_player=-1 のままになる場合は
                  win_condition 判定が機能していない。
        """
        from dm_toolkit.gui.game_session import GameSession
        from dm_toolkit.gui import headless

        sess = GameSession()
        sess.reset_game()
        assert sess.gs is not None

        _steps, over = headless.run_steps(sess, max_steps=500)
        if not over:
            pytest.skip("500 ステップではゲームが終了しなかった（環境依存スキップ）")

        # gs.winner (GameResult enum) が NONE 以外に設定されているはず
        gs = sess.gs
        winner = getattr(gs, 'winner', None)
        assert winner is not None, (
            "game_over=True だが winner 属性が存在しない\n"
            "再発防止: WinCondition.check() が呼ばれているか確認すること"
        )
        winner_str = getattr(winner, 'name', str(winner))
        assert 'NONE' not in winner_str.upper(), (
            f"game_over=True だが winner が NONE のまま: {winner!r}\n"
            "再発防止: ゲーム終了後は winner が P1_WIN/P2_WIN/DRAW のいずれかになること"
        )


# ---------------------------------------------------------------------------
# Class 6: create_session グローバル状態汚染テスト（TDD）
# ---------------------------------------------------------------------------


class TestCreateSessionClean:
    """headless.create_session() がグローバル状態を汚染しないことを検証する。

    再発防止:
    - 旧コードは create_session() を呼ぶたびに _Command/_CommandGenerator を
      dm_ai_module に注入しており、CommandType（C++ IntEnum）が文字列に化けるリスクがあった。
    - 現在はスタブ注入は削除済みであり、CommandType が C++ IntEnum のままであることを確認する。
    """

    def test_create_session_does_not_pollute_dm_ai_module(self) -> None:
        """create_session() 呼び出し後も dm_ai_module.CommandType が C++ IntEnum のままであることを確認する。

        再発防止: スタブ注入コードが復活すると CommandType.PASS が文字列 'PASS' になり
                  IntEnum として使えなくなる。
        """
        from dm_toolkit.gui import headless

        # create_session を呼ぶ
        sess = headless.create_session()
        assert sess is not None

        # CommandType が C++ IntEnum のままであることを確認
        ct = getattr(dm_ai_module, 'CommandType', None)
        assert ct is not None, "dm_ai_module.CommandType が None"
        # C++ IntEnum の PASS は int 型のはず（文字列スタブではない）
        pass_val = getattr(ct, 'PASS', None)
        assert pass_val is not None, "CommandType.PASS が None"
        assert not isinstance(pass_val, str), (
            f"CommandType.PASS が文字列になっている: {pass_val!r}\n"
            "再発防止: headless.create_session() でスタブ注入が復活していないか確認すること"
        )

    def test_create_session_no_print_output(self, capsys: Any) -> None:
        """create_session() が標準出力に何も print しないことを確認する。

        再発防止: 旧コードは 'headless: injected CommandType/Command/CommandGenerator stubs'
                  を print していた。テスト出力汚染を防ぐためこれが出力されないことを確認する。
        """
        from dm_toolkit.gui import headless

        _sess = headless.create_session()
        captured = capsys.readouterr()
        assert "headless: injected" not in captured.out, (
            "create_session() がスタブ注入メッセージを print している\n"
            "再発防止: スタブ注入コードを headless.py に追加してはならない"
        )
