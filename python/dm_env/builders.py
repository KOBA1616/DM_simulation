# python/dm_env/builders.py
"""CommandDef 構築ヘルパー。

【設計方針】
- 全関数は dm_ai_module.CommandDef インスタンスを返す。
- 辞書・文字列を返す実装は禁止（レガシーAction混入防止）。
- GUI の game_session.py および CLI の repl.py 双方から使用する。
- 再発防止: builders.py を経由せず CommandDef を直接組み立てるコードを
  増やさないこと。変更は必ずこのファイルに集中させる。
- 再発防止: PyQt6 / PySide6 の import は絶対に追加しないこと（ヘッドレス層汚染禁止）。
"""
from __future__ import annotations
from typing import Optional, Any
from python.dm_env._native import get_module


def make_mana_charge(source_instance_id: int) -> Any:
    """マナチャージコマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.MANA_CHARGE
    cmd.source_instance_id = source_instance_id
    return cmd


def make_play_card(
    source_instance_id: int,
    target_instance_id: Optional[int] = None,
    target_player_id: Optional[int] = None,
) -> Any:
    """カードプレイコマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PLAY_CARD
    cmd.source_instance_id = source_instance_id
    if target_instance_id is not None:
        cmd.target_instance_id = target_instance_id
    if target_player_id is not None:
        cmd.target_player_id = target_player_id
    return cmd


def make_attack_player(source_instance_id: int, target_player_id: int) -> Any:
    """プレイヤーへの攻撃コマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_PLAYER
    cmd.source_instance_id = source_instance_id
    cmd.target_player_id = target_player_id
    return cmd


def make_attack_creature(
    source_instance_id: int, target_instance_id: int
) -> Any:
    """クリーチャーへの攻撃コマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_CREATURE
    cmd.source_instance_id = source_instance_id
    cmd.target_instance_id = target_instance_id
    return cmd


def make_pass() -> Any:
    """パスコマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PASS
    return cmd


def make_use_shield_trigger(source_instance_id: int) -> Any:
    """シールドトリガー使用コマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.USE_SHIELD_TRIGGER
    cmd.source_instance_id = source_instance_id
    return cmd
