# -*- coding: utf-8 -*-
# Localized Japanese text.
from typing import Any, Dict, List, Optional, Union
import os
import json
from types import ModuleType

# dm_ai_module may be an optional compiled module; annotate as Optional[ModuleType]
m: Optional[ModuleType] = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    # leave m as None if module not available
    pass

def load_translations():
    """Load translations from JSON file."""
    # Resolve path relative to this file or project root
    # Assuming this file is in dm_toolkit/gui/
    # data/locales/ja.json is in project root/data/locales/

    # Try finding project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # dm_toolkit/gui -> ../../ -> project root
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    locale_path = os.path.join(project_root, 'data', 'locales', 'ja.json')

    if os.path.exists(locale_path):
        try:
            with open(locale_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading translations from {locale_path}: {e}")
            return {}
    return {}

# Load base translations from JSON
TRANSLATIONS: Dict[Any, str] = load_translations()

# Add Enum mappings if module is available
if m:
    # EffectActionType (only if available on the module)
    if hasattr(m, 'EffectActionType'):
        _effect_map = {
            'GRANT_KEYWORD': "キーワード付与",
            'MOVE_CARD': "カード移動",
            'FRIEND_BURST': "フレンド・バースト",
            'APPLY_MODIFIER': "効果付与",
            'DRAW_CARD': "カードを引く",
            'ADD_MANA': "マナ追加",
            'DESTROY': "破壊",
            'RETURN_TO_HAND': "手札に戻す",
            'SEND_TO_MANA': "マナ送りにする",
            'TAP': "タップする",
            'UNTAP': "アンタップする",
            'MODIFY_POWER': "パワー修正",
            'BREAK_SHIELD': "シールドブレイク",
            'LOOK_AND_ADD': "見て加える(サーチ)",
            'SEARCH_DECK_BOTTOM': "デッキ下サーチ",
            'MEKRAID': "メクレイド",
            'REVOLUTION_CHANGE': "革命チェンジ",
            'COUNT_CARDS': "カードカウント",
            'GET_GAME_STAT': "ゲーム統計取得",
            'REVEAL_CARDS': "カード公開",
            'RESET_INSTANCE': "カード状態リセット",
            'REGISTER_DELAYED_EFFECT': "遅延効果登録",
            'SEARCH_DECK': "デッキ探索",
            'SHUFFLE_DECK': "デッキシャッフル",
            'ADD_SHIELD': "シールド追加",
            'SEND_SHIELD_TO_GRAVE': "シールド焼却",
            'SEND_TO_DECK_BOTTOM': "デッキ下に送る",
            'MOVE_TO_UNDER_CARD': "カードの下に重ねる",
            'CAST_SPELL': "呪文を唱える",
            'PUT_CREATURE': "クリーチャーを出す",
            'COST_REFERENCE': "コスト参照/軽減",
            'SELECT_NUMBER': "数字を選択",
            'SUMMON_TOKEN': "トークン生成",
            'DISCARD': "手札を捨てる",
            'PLAY_FROM_ZONE': "ゾーンからプレイ",
            'LOOK_TO_BUFFER': "バッファへ移動(Look)",
            'SELECT_FROM_BUFFER': "バッファから選択",
            'PLAY_FROM_BUFFER': "バッファからプレイ",
            'MOVE_BUFFER_TO_ZONE': "バッファから移動",
            'SELECT_OPTION': "選択肢",
            'RESOLVE_BATTLE': "バトル解決",
            'IF': "IF判定",
            'IF_ELSE': "IF_ELSE判定",
            'ELSE': "ELSE判定",
        }
        for _name, _text in _effect_map.items():
            _member = getattr(m.EffectActionType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # ActionType (map by name to avoid referencing missing enum attributes)
    if hasattr(m, 'ActionType'):
        _action_map = {
            'PLAY_CARD': "カードをプレイ",
            'ATTACK_CREATURE': "クリーチャー攻撃",
            'ATTACK_PLAYER': "プレイヤー攻撃",
            'BLOCK': "ブロック",
            'USE_SHIELD_TRIGGER': "S・トリガー使用",
            'RESOLVE_EFFECT': "効果解決",
            'RESOLVE_PLAY': "プレイ解決",
            'DECLARE_PLAY': "プレイ宣言",
            'SELECT_TARGET': "対象選択",
            'USE_ABILITY': "能力使用",
            'DECLARE_REACTION': "リアクション宣言",
            'MANA_CHARGE': "マナゾーンに置く",
            'PAY_COST': "コスト支払い",
            'PASS': "パス",
        }
        for _name, _text in _action_map.items():
            _member = getattr(m.ActionType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # TriggerType
    if hasattr(m, 'TriggerType'):
        _trigger_map = {
            'ON_PLAY': "出た時 (CIP)",
            'ON_ATTACK': "攻撃する時",
            'ON_DESTROY': "破壊された時",
            'ON_OPPONENT_DRAW': "相手がドローした時",
            'S_TRIGGER': "S・トリガー",
            'TURN_START': "ターン開始時",
            'PASSIVE_CONST': "常在効果(パッシブ)",
        }
        for _name, _text in _trigger_map.items():
            _member = getattr(m.TriggerType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Civilization
    if hasattr(m, 'Civilization'):
        _civ_map = {
            'FIRE': "火",
            'WATER': "水",
            'NATURE': "自然",
            'LIGHT': "光",
            'DARKNESS': "闇",
            'ZERO': "無色",
        }
        for _name, _text in _civ_map.items():
            _member = getattr(m.Civilization, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Zone
    if hasattr(m, 'Zone'):
        _zone_map = {
            'HAND': "手札",
            'BATTLE': "バトルゾーン",
            'GRAVEYARD': "墓地",
            'MANA': "マナゾーン",
            'SHIELD': "シールドゾーン",
            'DECK': "デッキ",
            'BUFFER': "バッファ",
            'UNDER_CARD': "カードの下",
        }
        for _name, _text in _zone_map.items():
            _member = getattr(m.Zone, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # TargetScope
    if hasattr(m, 'TargetScope'):
        _ts_map = {
            'SELF': "自分",
            'TARGET_SELECT': "対象選択",
            'NONE': "なし",
        }
        for _name, _text in _ts_map.items():
            _member = getattr(m.TargetScope, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Command Types
    if hasattr(m, 'CommandType'):
        _cmd_map = {
            'TRANSITION': "カード移動",
            'MUTATE': "状態変更",
            'FLOW': "進行制御",
            'QUERY': "カード情報取得",
            'DECIDE': "決定",
            'DECLARE_REACTION': "リアクション宣言",
            'STAT': "統計更新",
            'GAME_RESULT': "ゲーム終了",
            'IF': "IF判定",
            'IF_ELSE': "IF_ELSE判定",
            'ELSE': "ELSE判定",
        }
        for _name, _text in _cmd_map.items():
            _member = getattr(m.CommandType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Flow Types
    if hasattr(m, 'FlowType'):
        _flow_map = {
            'PHASE_CHANGE': "フェーズ移行",
            'TURN_CHANGE': "ターン変更",
            'SET_ACTIVE_PLAYER': "手番変更",
        }
        for _name, _text in _flow_map.items():
            _member = getattr(m.FlowType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Mutation Types
    if hasattr(m, 'MutationType'):
        _mut_map = {
            'TAP': "タップ",
            'UNTAP': "アンタップ",
            'POWER_MOD': "パワー修正",
            'ADD_KEYWORD': "キーワード付与",
            'REMOVE_KEYWORD': "キーワード削除",
            'ADD_PASSIVE_EFFECT': "パッシブ効果付与",
            'ADD_COST_MODIFIER': "コスト修正付与",
            'ADD_PENDING_EFFECT': "待機効果追加",
        }
        for _name, _text in _mut_map.items():
            _member = getattr(m.MutationType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Stat Types
    if hasattr(m, 'StatType'):
        _stat_map = {
            'CARDS_DRAWN': "ドロー枚数",
            'CARDS_DISCARDED': "手札破棄枚数",
            'CREATURES_PLAYED': "クリーチャープレイ数",
            'SPELLS_CAST': "呪文詠唱数",
        }
        for _name, _text in _stat_map.items():
            _member = getattr(m.StatType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Game Result
    if hasattr(m, 'GameResult'):
        _gr_map = {
            'NONE': "なし",
            'P1_WIN': "P1勝利",
            'P2_WIN': "P2勝利",
            'DRAW': "引き分け",
        }
        for _name, _text in _gr_map.items():
            _member = getattr(m.GameResult, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Also keep string keys for Enums for backward compatibility or serialization
    enum_candidates = [
        getattr(m, 'ActionType', None),
        getattr(m, 'EffectActionType', None),
        getattr(m, 'TriggerType', None),
        getattr(m, 'Civilization', None),
        getattr(m, 'Zone', None),
        getattr(m, 'TargetScope', None),
        getattr(m, 'CommandType', None),
        getattr(m, 'FlowType', None),
        getattr(m, 'MutationType', None),
        getattr(m, 'StatType', None),
        # Note: Avoid adding GameResult string keys to prevent 'DRAW' from overriding group translation.
    ]
    for enum_cls in [e for e in enum_candidates if e is not None]:
        for member in enum_cls.__members__.values():
            if member in TRANSLATIONS:
                TRANSLATIONS[member.name] = TRANSLATIONS[member]

def translate(key: Any) -> str:
    """Return localized text when available, otherwise echo the key."""
    # Try direct lookup (works for Enums and strings)
    res = TRANSLATIONS.get(key)
    if res is not None:
        return res

    # If key is an Enum, try looking up its name (fallback)
    if hasattr(key, "name"):
         res = TRANSLATIONS.get(key.name)
         if res is not None:
             return res

    # If key is a string and not found, return as is
    return str(key)

def tr(text: Any) -> str:
    return translate(text)

def get_card_civilizations(card_data: Any) -> List[str]:
    """
    Returns a list of civilization names (e.g. ["FIRE", "NATURE"]) from card data.
    Handles C++ pybind11 objects and legacy dicts.
    """
    if not card_data:
        return ["COLORLESS"]

    if hasattr(card_data, 'civilizations') and card_data.civilizations:
        civs = []
        for c in card_data.civilizations:
            if hasattr(c, 'name'):
                civs.append(c.name)
            else:
                civs.append(str(c).split('.')[-1])
        return civs

    elif hasattr(card_data, 'civilization'):
        # Legacy singular
        c = card_data.civilization
        if hasattr(c, 'name'):
            return [c.name]
        return [str(c).split('.')[-1]]

    return ["COLORLESS"]

def get_card_civilization(card_data: Any) -> str:
    """
    Returns the primary civilization name as a string.
    If multiple, returns the first one.
    """
    civs = get_card_civilizations(card_data)
    if civs:
        return civs[0]
    return "COLORLESS"

def get_card_name_by_instance(game_state: Any, card_db: Dict[int, Any], instance_id: int) -> str:
    if not game_state or not m: return f"Inst_{instance_id}"

    try:
        # Assuming GameState has get_card_instance exposed
        inst = game_state.get_card_instance(instance_id)
        if inst:
            card_id = inst.card_id
            if card_id in card_db:
                return card_db[card_id].name  # type: ignore
    except Exception:
        pass

    return f"Inst_{instance_id}"

def describe_command(cmd: Any, game_state: Any, card_db: Any) -> str:
    """Generate a localized string description for a GameCommand."""
    if not m:
        return "GameCommand（ネイティブモジュール未ロード）"

    cmd_type = cmd.get_type()

    if cmd_type == m.CommandType.TRANSITION:
        # TransitionCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.card_instance_id)
        return f"[{tr('TRANSITION')}] {name} (P{c.owner_id}): {tr(c.from_zone)} -> {tr(c.to_zone)}"

    elif cmd_type == m.CommandType.MUTATE:
        # MutateCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.target_instance_id)
        mutation = tr(c.mutation_type)
        val = ""
        if c.mutation_type == m.MutationType.POWER_MOD:
            val = f"{c.int_value:+}"
        elif c.mutation_type == m.MutationType.ADD_KEYWORD:
            val = c.str_value

        return f"[{tr('MUTATE')}] {name}: {mutation} {val}".strip()

    elif cmd_type == m.CommandType.FLOW:
        # FlowCommand
        c = cmd
        flow = tr(c.flow_type)
        val = c.new_value
        if c.flow_type == m.FlowType.PHASE_CHANGE:
            # Cast int to Phase enum if possible
            try:
                val = tr(m.Phase(c.new_value))
            except:
                pass
        return f"[{tr('FLOW')}] {flow}: {val}"

    elif cmd_type == m.CommandType.QUERY:
        c = cmd
        return f"[{tr('QUERY')}] {tr(c.query_type)}"

    elif cmd_type == m.CommandType.DECIDE:
        c = cmd
        return f"[{tr('DECIDE')}] 選択肢: {c.selected_option_index}, 対象数: {len(c.selected_indices)}"

    elif cmd_type == m.CommandType.STAT:
        c = cmd
        return f"[{tr('STAT')}] {tr(c.stat)} += {c.amount}"

    elif cmd_type == m.CommandType.GAME_RESULT:
        c = cmd
        return f"[{tr('GAME_RESULT')}] {tr(c.result)}"

    return f"未対応コマンド: {cmd_type}"
