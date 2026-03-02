# tests/test_command_flow.py
"""CommandDef フロー結合テスト。

全テストは以下の単一経路のみ使用する:
  IntentGenerator.generate_legal_commands() → CommandDef リスト
  → game_instance.resolve_command(cmd) → GameState 検証

再発防止: map_action / action_to_command / execute_action は使用禁止。execute_command を使用すること。
再発防止: dm_toolkit.commands 等の旧レイヤーを呼び出してはならない。
"""
from __future__ import annotations
import pytest

dm_ai_module = pytest.importorskip("dm_ai_module", reason="Requires native engine")


def _is_native() -> bool:
    return bool(getattr(dm_ai_module, "IS_NATIVE", False))


pytestmark = pytest.mark.skipif(
    not _is_native(), reason="Requires native dm_ai_module (IS_NATIVE=True)"
)


def _fresh_game():  # type: ignore[return]
    """テスト用ゲームインスタンスを生成し、MANA フェーズまで進める。
    再発防止: GameInstance は (seed, db) のシグネチャ。db なしでは TypeError になる。
    再発防止: START_OF_TURN フェーズは合法コマンドが 0 件。MANA 以降まで進める必要がある。
    """
    db = dm_ai_module.CardDatabase()
    game = dm_ai_module.GameInstance(0, db)
    game.start_game()
    # START_OF_TURN → DRAW → MANA へ進める（MANA では PASS が生成される）
    for _ in range(2):
        try:
            dm_ai_module.PhaseManager.next_phase(game.state, db)
        except Exception:
            break
    return game, db


class TestLegalCommandGeneration:
    def test_returns_list_of_command_defs(self) -> None:
        game, db = _fresh_game()
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
        assert isinstance(legal, list)
        for cmd in legal:
            assert hasattr(cmd, "type"), f"CommandDef に type 属性がない: {cmd!r}"

    def test_pass_is_always_available(self) -> None:
        game, db = _fresh_game()
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
        types = [str(cmd.type) for cmd in legal]
        assert any("PASS" in t for t in types), (
            f"PASS コマンドが生成されなかった。types={types}"
        )


class TestCommandExecution:
    def test_resolve_command_no_exception(self) -> None:
        game, db = _fresh_game()
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(game.state, db)
        assert legal, "合法コマンドが空 — generate_legal_commands が PASS すら返さない"
        game.resolve_command(legal[0])  # 例外が出なければ合格

    def test_full_game_no_crash(self) -> None:
        """500 手まで実行してクラッシュしないことを確認する。"""
        game, db = _fresh_game()
        for _ in range(500):
            if getattr(game.state, "game_over", False):
                break
            legal = dm_ai_module.IntentGenerator.generate_legal_commands(
                game.state, db
            )
            if not legal:
                break
            game.resolve_command(legal[0])
        # クラッシュしなければ合格
