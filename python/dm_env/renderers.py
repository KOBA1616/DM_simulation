# python/dm_env/renderers.py
"""GameState のターミナル可視化レンダラー。

PyQt6 禁止。print() ベース軽量実装。
rich ライブラリがあればリッチ表示、なければ plain text fallback。
再発防止: PyQt6 / PySide6 の import を絶対に追加しないこと。
"""
from __future__ import annotations
from typing import Any, List

try:
    from rich.console import Console as _Console
    _console = _Console()
    _RICH = True
except ImportError:
    _RICH = False


def render_game_state(state: Any) -> None:
    """GameState をターミナルに描画する。"""
    sep = "=" * 60
    print(sep)
    print(f"Turn: {state.turn_number} | Phase: {state.current_phase}")
    print(f"Active Player: Player {state.active_player_id}")
    print("-" * 60)
    for pid, player in enumerate(state.players):
        label = "（あなた）" if pid == state.active_player_id else "（相手）"
        hand  = getattr(player, "hand", [])
        mana  = getattr(player, "mana_zone", [])
        bz    = getattr(player, "battle_zone", [])
        print(f"\n  Player {pid} {label}")
        print(f"  Hand({len(hand)}): {_fmt_cards(hand)}")
        print(f"  Mana({len(mana)}): {_fmt_cards(mana)}")
        print(f"  BattleZone({len(bz)}): {_fmt_cards(bz)}")
    print(sep)


def render_legal_commands(commands: List[Any]) -> None:
    """合法コマンドを選択肢形式で表示する。"""
    print("\n=== コマンドを選択 ===")
    for i, cmd in enumerate(commands):
        print(f"  [{i + 1}] {_fmt_command(cmd)}")
    print("> ", end="", flush=True)


def _fmt_cards(cards: List[Any]) -> str:
    parts = []
    for c in cards[:6]:
        iid  = getattr(c, "instance_id", "?")
        name = getattr(c, "name", f"#{iid}")
        parts.append(f"{name}(#{iid})")
    if len(cards) > 6:
        parts.append(f"…+{len(cards)-6}")
    return ", ".join(parts) or "（なし）"


def _fmt_command(cmd: Any) -> str:
    ctype = str(getattr(cmd, "type", "UNKNOWN"))
    # 再発防止: CommandDef フィールドは instance_id / target_instance。
    # source_instance_id / target_instance_id は存在しない（bind_core.cpp 参照）。
    src  = getattr(cmd, "instance_id", -1)
    tgt  = getattr(cmd, "target_instance", -1)
    if tgt >= 0:
        return f"{ctype}  src=#{src} → target=#{tgt}"
    return f"{ctype}  src=#{src}"
