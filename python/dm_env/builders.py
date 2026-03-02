# python/dm_env/builders.py
"""CommandDef 構築ヘルパー。

【設計方針】
- 全関数は dm_ai_module.CommandDef インスタンスを返す。
- 辞書・文字列を返す実装は禁止（レガシーAction混入防止）。
- GUI の game_session.py および CLI の repl.py 双方から使用する。
- 再発防止: builders.py を経由せず CommandDef を直接組み立てるコードを
  増やさないこと。変更は必ずこのファイルに集中させる。
- 再発防止: PyQt6 / PySide6 の import は絶対に追加しないこと（ヘッドレス層汚染禁止）。

【CommandDef フィールド対応表】 (bind_core.cpp 実測)
  instance_id       : 操作元カードの instance_id
  target_instance   : 対象カードの instance_id（クリーチャー攻撃先など）
  owner_id          : 操作プレイヤーID
  slot_index        : 手札スロットインデックス
  amount            : 0=クリーチャー面, 1=呪文面（PLAY_FROM_ZONE のみ）

【合法コマンド生成】
  dm.IntentGenerator.generate_legal_commands(state, card_db)
  ※ generate_legal_actions は後方互換エイリアス（新規コードでは使用禁止）。
"""
from __future__ import annotations
from typing import Optional, Any
from python.dm_env._native import get_module


def make_mana_charge(instance_id: int, slot_index: int = 0) -> Any:
    """マナチャージコマンドを返す。

    Args:
        instance_id: 手札カードの instance_id
        slot_index:  手札スロットインデックス
    """
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.MANA_CHARGE
    cmd.instance_id = instance_id
    cmd.slot_index = slot_index
    return cmd


def make_play_card(
    instance_id: int,
    slot_index: int = 0,
    is_spell_side: bool = False,
    target_instance: Optional[int] = None,
) -> Any:
    """カードプレイコマンドを返す（PLAY_FROM_ZONE）。

    Args:
        instance_id:   手札カードの instance_id
        slot_index:    手札スロットインデックス
        is_spell_side: True=ツインパクト呪文面, False=クリーチャー面
        target_instance: 対象クリーチャーの instance_id（呪文等）
    """
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PLAY_FROM_ZONE
    cmd.instance_id = instance_id
    cmd.slot_index = slot_index
    cmd.amount = 1 if is_spell_side else 0
    if target_instance is not None:
        cmd.target_instance = target_instance
    return cmd


def make_attack_player(instance_id: int, slot_index: int = 0) -> Any:
    """プレイヤーへの攻撃コマンドを返す。

    Args:
        instance_id: 攻撃クリーチャーの instance_id
        slot_index:  battle_zone スロットインデックス
    """
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_PLAYER
    cmd.instance_id = instance_id
    cmd.target_instance = -1  # C++ 側は -1 でプレイヤー攻撃を示す
    cmd.slot_index = slot_index
    return cmd


def make_attack_creature(
    instance_id: int, target_instance: int, slot_index: int = 0
) -> Any:
    """クリーチャーへの攻撃コマンドを返す。

    Args:
        instance_id:     攻撃クリーチャーの instance_id
        target_instance: 防御クリーチャーの instance_id
        slot_index:      攻撃クリーチャーの battle_zone スロット
    """
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.ATTACK_CREATURE
    cmd.instance_id = instance_id
    cmd.target_instance = target_instance
    cmd.slot_index = slot_index
    return cmd


def make_pass() -> Any:
    """パスコマンドを返す。"""
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.PASS
    return cmd


def make_use_shield_trigger(instance_id: int) -> Any:
    """シールドトリガー使用コマンドを返す（SHIELD_TRIGGER）。

    Args:
        instance_id: トリガー発動カードの instance_id
    """
    dm = get_module()
    cmd = dm.CommandDef()
    cmd.type = dm.CommandType.SHIELD_TRIGGER
    cmd.instance_id = instance_id
    return cmd
