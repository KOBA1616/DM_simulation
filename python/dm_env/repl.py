# python/dm_env/repl.py
"""CLI インタラクティブ REPL（人間対AI）。

標準入力からの番号選択のみ使用。PyQt6 不要。
再発防止: GUI依存（PyQt6等）を追加しないこと。
"""
from __future__ import annotations
import random
from typing import List, Any

from python.dm_env._native import get_module
from python.dm_env.renderers import render_game_state, render_legal_commands

MAX_TURNS: int = 300


def run_interactive(
    deck_p0: List[int],
    deck_p1: List[int],
    seed: int | None = None,
) -> None:
    """ターミナルで人間がプレイするメインループ。
    Player 0 が人間、Player 1 が SimpleAI（先頭コマンド選択）。
    """
    dm = get_module()
    _seed = seed if seed is not None else random.randint(0, 999999)
    db = dm.CardDatabase()
    game = dm.GameInstance(_seed, db)
    game.start_game()

    for _ in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            winner = getattr(state, "winner", "?")
            print(f"\n===== ゲーム終了 | 勝者: Player {winner} =====")
            return

        render_game_state(state)
        legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db)

        if not legal:
            print("[警告] 合法コマンドなし。ゲームを終了します。")
            break

        if state.active_player_id == 0:
            render_legal_commands(legal)
            idx = _read_choice(len(legal))
            cmd = legal[idx] if idx < len(legal) else _pass_cmd(dm)
        else:
            # SimpleAI: 先頭を選択（MCTSへの差し替えポイント）
            cmd = legal[0]

        game.resolve_command(cmd)

    print("[MaxTurns] ゲームが上限ターンに達しました。")


def _read_choice(n: int) -> int:
    while True:
        try:
            val = int(input())
            if 1 <= val <= n:
                return val - 1   # 1-indexed → 0-indexed
            print(f"1〜{n} の番号を入力 > ", end="", flush=True)
        except (ValueError, EOFError):
            return n   # 範囲外 → PASS 扱い（illegal index → last+1 → PASS）


def _pass_cmd(dm: Any) -> Any:
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PASS
    return cmd
