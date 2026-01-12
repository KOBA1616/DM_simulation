# -*- coding: utf-8 -*-
import json
import os
from typing import Any, Dict, Optional

# dm_ai_module
m: Optional[Any] = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    pass

TRANSLATIONS: Dict[Any, str] = {}

def load_translations():
    global TRANSLATIONS
    # Load JSON
    # Resolve absolute path relative to this file
    # This file is in dm_toolkit/gui/i18n.py
    # Root is ../../
    json_path = os.path.join(os.path.dirname(__file__), "../../data/locale/ja.json")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            TRANSLATIONS.update(json.load(f))
    except FileNotFoundError:
        print(f"Warning: Translation file not found at {json_path}")

    # Add Enums if module is available
    if m:
        # Helper to set enum translation only if not already present (or force update? No, we prefer JSON)
        # But wait, the JSON keys are strings (e.g. "GRANT_KEYWORD").
        # The runtime keys are Enums (e.g. EffectActionType.GRANT_KEYWORD).
        # We need to map the Enum key to the text.
        # The text source should be: JSON if available for the Enum name, else hardcoded default.

        def register_enum_translations(enum_cls, name_map):
            if not enum_cls: return
            for _name, _default_text in name_map.items():
                _member = getattr(enum_cls, _name, None)
                if _member is not None:
                    # Check if JSON has a translation for the string name
                    if _name in TRANSLATIONS:
                        TRANSLATIONS[_member] = TRANSLATIONS[_name]
                    # Or maybe the JSON has the default text as key? No, JSON has keys.
                    # The JSON file contains "GRANT_KEYWORD": "キーワード付与"
                    # So TRANSLATIONS["GRANT_KEYWORD"] exists.
                    # We want TRANSLATIONS[EffectActionType.GRANT_KEYWORD] = "キーワード付与"

                    # If the name is in TRANSLATIONS, use that value.
                    # If not, use the hardcoded default.

                    if _name in TRANSLATIONS:
                        TRANSLATIONS[_member] = TRANSLATIONS[_name]
                    else:
                        TRANSLATIONS[_member] = _default_text

        # EffectActionType
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
            register_enum_translations(m.EffectActionType, _effect_map)

        # ActionType
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
            register_enum_translations(m.ActionType, _action_map)

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
            register_enum_translations(m.TriggerType, _trigger_map)

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
            register_enum_translations(m.Civilization, _civ_map)

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
            register_enum_translations(m.Zone, _zone_map)

        # TargetScope
        if hasattr(m, 'TargetScope'):
            _ts_map = {
                'SELF': "自分",
                'TARGET_SELECT': "対象選択",
                'NONE': "なし",
            }
            register_enum_translations(m.TargetScope, _ts_map)

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
            register_enum_translations(m.CommandType, _cmd_map)

        # Flow Types
        if hasattr(m, 'FlowType'):
            _flow_map = {
                'PHASE_CHANGE': "フェーズ移行",
                'TURN_CHANGE': "ターン変更",
                'SET_ACTIVE_PLAYER': "手番変更",
            }
            register_enum_translations(m.FlowType, _flow_map)

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
            register_enum_translations(m.MutationType, _mut_map)

        # Stat Types
        if hasattr(m, 'StatType'):
            _stat_map = {
                'CARDS_DRAWN': "ドロー枚数",
                'CARDS_DISCARDED': "手札破棄枚数",
                'CREATURES_PLAYED': "クリーチャープレイ数",
                'SPELLS_CAST': "呪文詠唱数",
            }
            register_enum_translations(m.StatType, _stat_map)

        # Game Result
        if hasattr(m, 'GameResult'):
            _gr_map = {
                'NONE': "なし",
                'P1_WIN': "P1勝利",
                'P2_WIN': "P2勝利",
                'DRAW': "引き分け",
            }
            register_enum_translations(m.GameResult, _gr_map)

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
        ]
        for enum_cls in [e for e in enum_candidates if e is not None]:
            for member in enum_cls.__members__.values():
                if member in TRANSLATIONS:
                    TRANSLATIONS[member.name] = TRANSLATIONS[member]

# Initialize
load_translations()

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
