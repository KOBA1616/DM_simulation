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


def _serialize_winner(winner: Any) -> Optional[int]:
    """GameResult を JSON シリアライズ可能な int/None に変換する。

    再発防止: C++ GameResult enum は json.dump できないため必ずこの関数を通す。
    - GameResult.NONE  → None（ゲーム継続中）
    - GameResult.P1_WIN → 0
    - GameResult.P2_WIN → 1
    - GameResult.DRAW   → -1
    """
    if winner is None:
        return None
    try:
        # C++ enum: str 表現で判定
        name = getattr(winner, "name", str(winner))
        if "NONE" in name:
            return None
        if "P1_WIN" in name:
            return 0
        if "P2_WIN" in name:
            return 1
        if "DRAW" in name:
            return -1
        # fallback: int へ強制変換
        return int(winner)
    except Exception:
        return None


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
    # 初期化: 可能ならネイティブの PhaseManager.start_game を使って
    # シールドや手札の初期配置を確実に行う（GameInstance.start_game のみでは不足する場合がある）
    # デッキ指定がある場合は state にセットする（空リストは無視）
    try:
        gs = game.state
        if deck_p0:
            try:
                gs.set_deck(0, deck_p0)
            except Exception:
                pass
        else:
            # デフォルトデッキ（テスト用）
            try:
                gs.set_deck(0, [1] * 40)
            except Exception:
                pass
        if deck_p1:
            try:
                gs.set_deck(1, deck_p1)
            except Exception:
                pass
        else:
            try:
                gs.set_deck(1, [1] * 40)
            except Exception:
                pass

    except Exception:
        # state が未設定でも進める
        pass

    try:
        pm = getattr(dm, "PhaseManager", None)
        if pm is not None and hasattr(pm, "start_game"):
            # 診断: PhaseManager による初期化をトレース
            try:
                pm.start_game(game.state, db)
                # ログは後続の処理で返却するため、一時的に記録しておく
                init_log = ["[init] PhaseManager.start_game invoked"]
            except Exception as exc:
                init_log = [f"[init] PhaseManager.start_game error: {exc}"]
                # フォールバックで game.start_game を試す
                try:
                    game.start_game()
                    init_log.append("[init] fallback GameInstance.start_game succeeded")
                except Exception as exc2:
                    init_log.append(f"[init] fallback GameInstance.start_game error: {exc2}")
                    return {"winner": None, "turns": 0, "log": init_log}
        else:
            game.start_game()
            init_log = ["[init] GameInstance.start_game invoked"]
    except Exception as exc:
        return {"winner": None, "turns": 0, "log": [f"start_game error: {exc}"]}
    log: List[str] = []
    turn = 0

    # 初期化ログを先頭に置く
    log: List[str] = init_log if 'init_log' in locals() else []
    for turn in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            break
        try:
            # Prefer toolkit wrapper which includes fast_forward and python fallbacks
            try:
                from dm_toolkit.commands import generate_legal_commands
                legal: List[Any] = generate_legal_commands(state, db, strict=False, skip_wrapper=False) or []
            except Exception:
                legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db) or []
        except Exception as exc:
            log.append(f"[Turn {turn}] generate_legal_commands error: {exc}")
            break
        if not legal:
            # 合法手が見つからない場合、初期化不足の可能性があるため
            # PhaseManager.start_game() を試みて再生成する（1度だけフォールバック）
            pm = getattr(dm, "PhaseManager", None)
            if pm is not None and hasattr(pm, "start_game"):
                try:
                    pm.start_game(game.state, db)
                    try:
                        from dm_toolkit.commands import generate_legal_commands
                        legal = generate_legal_commands(state, db, strict=False, skip_wrapper=False) or []
                    except Exception:
                        legal = dm.IntentGenerator.generate_legal_commands(state, db) or []
                except Exception as exc:
                    log.append(f"[Turn {turn}] reinit generate_legal_commands error: {exc}")
            if not legal:
                # 追加診断情報を収集
                try:
                    s = game.state
                    ap = getattr(s, 'active_player_id', None)
                    diag = [f"[diag] active_player_id={ap}"]
                    players = getattr(s, 'players', None)
                    if players is not None:
                        for pid, p in enumerate(players):
                            try:
                                deck_len = len(getattr(p, 'deck', []))
                            except Exception:
                                deck_len = 'NA'
                            try:
                                hand_len = len(getattr(p, 'hand', []))
                            except Exception:
                                hand_len = 'NA'
                            try:
                                shield_len = len(getattr(p, 'shield_zone', []))
                            except Exception:
                                shield_len = 'NA'
                            diag.append(f"[diag] P{pid}: deck={deck_len} hand={hand_len} shields={shield_len}")
                    else:
                        diag.append(f"[diag] players attribute missing")
                    log.extend(diag)
                except Exception:
                    log.append("[diag] failed to collect state diagnostics")
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
        # 再発防止: GameResult は C++ enum で JSON 非シリアライザブルのため int/None に変換
        "winner": _serialize_winner(getattr(game.state, "winner", None)),
        "turns": turn,
        "log": log,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
