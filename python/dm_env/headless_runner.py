# python/dm_env/headless_runner.py
"""ヘッドレスゲームループ（AI対AI・バッチシミュレーション用）。

PyQt6 import 禁止。JSON ログ出力オプション付き。
再発防止: PyQt6 を import するとヘッドレス CI が壊れる。絶対に追加禁止。
"""
from __future__ import annotations
import json
import os
import random
from typing import Any, Dict, List, Optional
from python.dm_env._native import get_module

_CARDS_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cards.json")

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
    # 再発防止: CardDatabase() は空DBを返すため cards.json をロードすること。
    # GameSession は JsonLoader.load_cards() を使用しており整合性を保つ必要がある。
    try:
        cards_path = os.path.normpath(_CARDS_JSON)
        if hasattr(dm, "JsonLoader") and os.path.exists(cards_path):
            db = dm.JsonLoader.load_cards(cards_path)
        else:
            db = dm.CardDatabase()
    except Exception:
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
        # NOTE: 再発防止 — 初期化は必ず game.start_game() を使うこと。
        # PhaseManager.start_game は Phase 管理専用であり、手札・シールドの初期配置を行わない。
        # PhaseManager.start_game のみ呼ぶとデッキ/手札が未設定のままゲームが始まり
        # actions=0 / turns=1 の不整合が発生する。
        game.start_game()
        init_log = ["[init] GameInstance.start_game invoked"]
    except Exception as exc:
        return {"winner": None, "turns": 0, "log": [f"start_game error: {exc}"]}
    # 再発防止: turn はアクション数(resolve_commandの呼出回数)であり "ゲームのターン数" ではない。
    # game_rounds でプレイヤーターン数（active_player_id の切り替わり回数）を別途計測する。
    action_count: int = 0
    game_rounds: int = 0
    _prev_active_player: Optional[int] = None

    # 初期化ログを先頭に置く
    log: List[str] = init_log if 'init_log' in locals() else []
    for _loop_idx in range(MAX_TURNS):
        state = game.state
        if state.game_over:
            break
        # プレイヤーターン数カウント: active_player_id が変わったら1ターン追加
        _cur_player = getattr(state, 'active_player_id', None)
        if _cur_player is not None and _cur_player != _prev_active_player:
            game_rounds += 1
            _prev_active_player = _cur_player
        try:
            # 再発防止: dm_toolkit.commands はラッパーオブジェクトを返すため
            # game.resolve_command に渡せない。必ずネイティブ IntentGenerator を使うこと。
            legal: List[Any] = dm.IntentGenerator.generate_legal_commands(state, db) or []
        except Exception as exc:
            log.append(f"[Action {action_count}] generate_legal_commands error: {exc}")
            break
        if not legal:
            # 再発防止: START_OF_TURN / DRAW フェーズは IntentGenerator が意図的に
            # 空リストを返す（自動進行フェーズ）。PhaseManager.fast_forward() でフェーズを
            # 進めてから再度 legal commands を生成する。
            # PhaseManager.start_game() の再呼び出しは禁止（ゲーム状態がリセットされる）。
            pm_cls = getattr(dm, "PhaseManager", None)
            if pm_cls is not None and hasattr(pm_cls, "fast_forward"):
                try:
                    pm_cls.fast_forward(game.state, db)
                    legal = dm.IntentGenerator.generate_legal_commands(game.state, db) or []
                    # fast_forward 後も空なら next_phase で再試行（最大1回）
                    if not legal and hasattr(pm_cls, "next_phase"):
                        pm_cls.next_phase(game.state, db)
                        legal = dm.IntentGenerator.generate_legal_commands(game.state, db) or []
                except Exception as exc:
                    log.append(f"[Action {action_count}] fast_forward error: {exc}")
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
                log.append(f"[Action {action_count}] No legal commands — 強制終了。")
                break
        # SimpleAI: 先頭コマンドを選択（MCTS等への差し替えポイント）
        cmd = legal[0]
        try:
            game.resolve_command(cmd)
        except Exception as exc:
            log.append(f"[Action {action_count}] resolve_command error: {exc}")
            break
        action_count += 1
        if verbose:
            log.append(
                f"[Action {action_count} / Round {game_rounds}]"
                f" P{state.active_player_id} → {getattr(cmd, 'type', '?')}"
            )
    else:
        log.append(f"[MaxActions {MAX_TURNS}] 上限アクション数に達しました。")

    result: Dict[str, Any] = {
        # 再発防止: GameResult は C++ enum で JSON 非シリアライザブルのため int/None に変換
        "winner": _serialize_winner(getattr(game.state, "winner", None)),
        # 再発防止: "turns" = プレイヤーターン数（active_player_id の切り替わり件数）。
        # 旧実装では resolve_command のループカウンタ（アクション数）を誤って "turns" としていた。
        # アクション数は "actions" キーで提供する。
        "turns": game_rounds,
        "actions": action_count,
        "log": log,
    }
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
