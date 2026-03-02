# python/dm_env/headless_runner.py
"""ヘッドレスゲームループ（AI対AI・バッチシミュレーション用）。

PyQt6 import 禁止。JSON ログ出力オプション付き。
再発防止: PyQt6 を import するとヘッドレス CI が壊れる。絶対に追加禁止。
"""
from __future__ import annotations
import json
import random
from typing import Any, Dict, List, Optional
from python.dm_env._native import get_module

MAX_TURNS: int = 300  # 無限ループ防止


def run_game(
    deck_p0: List[int],
    deck_p1: List[int],
    output_json: Optional[str] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """1ゲームを最初から最後まで実行し、結果を返す。

    Args:
        deck_p0: Player 0 のデッキ（カードID リスト）
        deck_p1: Player 1 のデッキ（カードID リスト）
        output_json: 指定した場合、結果をこのパスに JSON 出力する
        verbose: True の場合、各ターンのログを記録する
        seed: 再現性のための乱数シード（省略時はランダム）

    Returns:
        {"winner": int | None, "turns": int, "log": List[str]}
    """
    dm = get_module()
    _seed = seed if seed is not None else random.randint(0, 999999)
    db = dm.CardDatabase()
    game = dm.GameInstance(_seed, db)
    game.start_game()
    log: List[str] = []
    turn = 0

    for turn in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            break
        try:
            legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db)
        except Exception as exc:
            log.append(f"[Turn {turn}] generate_legal_commands error: {exc}")
            break
        if not legal:
            log.append(f"[Turn {turn}] No legal commands — 強制終了。")
            break
        # SimpleAI: 先頭コマンドを選択（MCTS等への差し替えポイント）
        cmd = legal[0]
        try:
            game.resolve_command(cmd)
        except Exception as exc:
            log.append(f"[Turn {turn}] resolve_command error: {exc}")
            break
        if verbose:
            log.append(
                f"[Turn {turn}] P{state.active_player_id} → {getattr(cmd, 'type', '?')}"
            )
    else:
        log.append(f"[MaxTurns {MAX_TURNS}] 上限ターンに達しました。")

    result: Dict[str, Any] = {
        "winner": getattr(game.state, "winner", None),
        "turns": turn,
        "log": log,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
