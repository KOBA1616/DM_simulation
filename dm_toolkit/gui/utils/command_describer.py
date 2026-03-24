# -*- coding: utf-8 -*-
from typing import Any, Dict, Tuple
from ..i18n import tr
from .card_helpers import get_card_name_by_instance
import logging

logger = logging.getLogger(__name__)

m: Any = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    pass

# 再発防止: type(cmd).__name__ は _CommandWrapper を返すため使用しない。
# cmd.to_dict()['type'] から CommandType を正規化して識別する。
# C++ CommandType enum の int 値 → 文字列マッピング
_CMD_INT_TO_STR: Dict[int, str] = {
    0: 'TRANSITION', 1: 'MUTATE', 2: 'FLOW', 3: 'QUERY',
    4: 'DRAW_CARD', 5: 'DISCARD', 6: 'DESTROY', 7: 'BOOST_MANA',
    8: 'TAP', 9: 'UNTAP', 10: 'BREAK_SHIELD', 11: 'SHIELD_TRIGGER',
    12: 'MOVE_CARD', 13: 'SEND_TO_MANA', 14: 'MANA_CHARGE', 15: 'MANA_CHARGE',
    16: 'ATTACK_PLAYER', 17: 'ATTACK_CREATURE', 18: 'BLOCK',
    19: 'PLAY_FROM_ZONE', 20: 'CAST_SPELL', 21: 'PASS',
    22: 'SELECT_TARGET', 23: 'CHOICE', 24: 'NONE',
}


def _get_cmd_type_str(cmd: Any) -> Tuple[str, dict]:
    """cmd から正規化済みコマンド種別文字列と to_dict() 結果を返す。
    _CommandWrapper / 生 CommandDef の両方に対応。
    """
    d: dict = {}
    try:
        d = cmd.to_dict()
        raw = d.get('type', '')
        # pybind11 バインドされた enum は .name 属性を持つ
        if hasattr(raw, 'name'):
            return raw.name, d
        # Python IntEnum: str() → "CommandType.PLAY_FROM_ZONE"
        s = str(raw).split('.')[-1].upper()
        if s.lstrip('-').isdigit():
            return _CMD_INT_TO_STR.get(int(s), f'CMD_{s}'), d
        return s, d
    except Exception:
        return type(cmd).__name__, d


def describe_command(cmd: Any, game_state: Any, card_db: Any) -> str:
    """Generate a localized string description for a GameCommand."""
    if not m:
        return "GameCommand（ネイティブモジュール未ロード）"

    cmd_type, d = _get_cmd_type_str(cmd)
    inst_id = d.get('instance_id')

    if cmd_type == 'PLAY_FROM_ZONE':
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id) or tr('Card')
            return f"[{tr('PLAY_CARD')}] {name}"
        except Exception:
            return f"[{tr('PLAY_CARD')}]"

    elif cmd_type in ('MANA_CHARGE', 'PLAYER_MANA_CHARGE', 'SEND_TO_MANA'):
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id)
            if name:
                return f"[{tr('MANA_CHARGE')}] {name}"
        except Exception:
            pass
        return f"[{tr('MANA_CHARGE')}]"

    elif cmd_type == 'CAST_SPELL':
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id) or tr('Card')
            return f"[{tr('CAST_SPELL')}] {name}"
        except Exception:
            return f"[{tr('CAST_SPELL')}]"

    elif cmd_type == 'ATTACK_PLAYER':
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id) or tr('Card')
            return f"[{tr('ATTACK_PLAYER')}] {name}"
        except Exception:
            return f"[{tr('ATTACK_PLAYER')}]"

    elif cmd_type == 'ATTACK_CREATURE':
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id) or tr('Card')
            return f"[{tr('ATTACK_CREATURE')}] {name}"
        except Exception:
            return f"[{tr('ATTACK_CREATURE')}]"

    elif cmd_type == 'PASS':
        return f"[{tr('Pass / End')}]"

    elif cmd_type == 'TRANSITION':
        try:
            from_zone = d.get('from_zone', '?')
            to_zone = d.get('to_zone', '?')
            name = get_card_name_by_instance(game_state, card_db, inst_id)
            if name:
                return f"[{tr('TRANSITION')}] {name}: {tr(str(from_zone))} → {tr(str(to_zone))}"
        except Exception:
            pass
        return f"[{tr('TRANSITION')}]"

    elif cmd_type == 'MUTATE':
        try:
            target_id = d.get('target_instance') or d.get('instance_id')
            name = get_card_name_by_instance(game_state, card_db, target_id)
            mutation_kind = d.get('mutation_kind', '?')
            if name:
                return f"[{tr('MUTATE')}] {name}: {tr(str(mutation_kind))}"
        except Exception:
            pass
        return f"[{tr('MUTATE')}]"

    elif cmd_type == 'FLOW':
        return f"[{tr('FLOW')}]"

    elif cmd_type == 'QUERY':
        return f"[{tr('QUERY')}]"

    elif cmd_type == 'DECIDE':
        return f"[{tr('DECIDE')}]"

    elif cmd_type == 'BLOCK':
        try:
            name = get_card_name_by_instance(game_state, card_db, inst_id) or tr('Card')
            return f"[ブロック] {name}"
        except Exception:
            return "[ブロック]"

    return f"[{cmd_type}]"
