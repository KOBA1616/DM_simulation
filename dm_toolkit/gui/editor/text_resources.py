# -*- coding: utf-8 -*-
"""
Centralized text resources for Static Abilities and Trigger Effects.
Maps trigger types, conditions, scopes, and modifier types to Japanese text.
"""

from typing import Dict, Optional, Tuple
from dm_toolkit.consts import TargetScope


class CardTextResources:
    """
    Centralized Japanese text resource library for card abilities.
    Used by TextGenerator, FormEditors, and UI components.
    """
    
    # Condition Japanese mapping (shared by both STATIC and TRIGGER contexts)
    CONDITION_JAPANESE: Dict[str, str] = {
        "NONE": "",
        "DURING_YOUR_TURN": "自分のターン中、",
        "DURING_OPPONENT_TURN": "相手のターン中、",
        "OPPONENT_DRAW_COUNT": "相手がカードを引いた時、"
    }
    
    # IF判定用の条件タイプの日本語化（GUIエディタ用）
    CONDITION_TYPE_LABELS: Dict[str, str] = {
        "NONE": "なし",
        "MANA_ARMED": "マナ武装",
        "SHIELD_COUNT": "シールド枚数",
        "CIVILIZATION_MATCH": "文明一致",
        "OPPONENT_PLAYED_WITHOUT_MANA": "相手がマナなしでプレイ",
        "OPPONENT_DRAW_COUNT": "相手のドロー枚数",
        "DURING_YOUR_TURN": "自分のターン中",
        "DURING_OPPONENT_TURN": "相手のターン中",
        "FIRST_ATTACK": "初回攻撃",
        "EVENT_FILTER_MATCH": "イベントフィルター一致",
        "COMPARE_STAT": "統計値比較",
        "COMPARE_INPUT": "入力値比較",
        "CARDS_MATCHING_FILTER": "フィルター一致カード数",
        "DECK_EMPTY": "デッキ切れ",
        "MANA_CIVILIZATION_COUNT": "マナゾーン文明数",
        "HAND_COUNT": "手札枚数",
        "BATTLE_ZONE_COUNT": "バトルゾーンカード数",
        "GRAVEYARD_COUNT": "墓地カード数",
        "CUSTOM": "カスタム"
    }
    
    # Scope/Owner Japanese mapping (STATIC ability context)
    # Now using TargetScope constants for consistency
    SCOPE_JAPANESE: Dict[str, str] = {
        TargetScope.SELF: "自分の",
        TargetScope.OPPONENT: "相手の",
        TargetScope.ALL: "",
        # Legacy support
        "SELF": "自分の",
        "OPPONENT": "相手の",
        "ALL": "",
        "PLAYER_SELF": "自分",
        "PLAYER_OPPONENT": "相手",
        "ALL_PLAYERS": "両プレイヤー",
    }
    
    # Trigger event Japanese mapping
    TRIGGER_JAPANESE: Dict[str, str] = {
        "ON_PLAY": "このクリーチャーが出た時",
        "ON_OTHER_ENTER": "他のクリーチャーが出た時",
        "AT_ATTACK": "このクリーチャーが攻撃する時",
        "ON_DESTROY": "このクリーチャーが破壊された時",
        "AT_END_OF_TURN": "自分のターンの終わりに",
        "AT_END_OF_OPPONENT_TURN": "相手のターンの終わりに",
        "ON_BLOCK": "このクリーチャーがブロックした時",
        "ON_ATTACK_FROM_HAND": "手札から攻撃する時",
        "TURN_START": "自分のターンのはじめに",
        "S_TRIGGER": "S・トリガー",
        "PASSIVE_CONST": "（常在効果）",
        "ON_SHIELD_ADD": "カードがシールドゾーンに置かれた時",
        "AT_BREAK_SHIELD": "シールドをブレイクする時",
        "ON_CAST_SPELL": "呪文を唱えた時",
        "ON_OPPONENT_DRAW": "相手がカードを引いた時",
        "NONE": ""
    }
    
    # Spell-specific trigger mapping
    SPELL_TRIGGER_JAPANESE: Dict[str, str] = {
        "ON_PLAY": "この呪文を唱えた時",
        "ON_OTHER_ENTER": "他のクリーチャーが出た時",
        # Other triggers remain same as creature
    }
    
    # Modifier type Japanese mapping
    MODIFIER_TYPE_JAPANESE: Dict[str, str] = {
        "NONE": "なし",
        "COST_MODIFIER": "コスト軽減",
        "POWER_MODIFIER": "パワー修正",
        "GRANT_KEYWORD": "キーワード付与",
        "SET_KEYWORD": "キーワード設定",
        "ADD_RESTRICTION": "制限追加"
    }
    
    # Keyword Japanese mapping
    KEYWORD_TRANSLATION: Dict[str, str] = {
        # Lowercase versions (original)
        "speed_attacker": "スピードアタッカー",
        "blocker": "ブロッカー",
        "slayer": "スレイヤー",
        "double_breaker": "W・ブレイカー",
        "triple_breaker": "T・ブレイカー",
        "world_breaker": "ワールド・ブレイカー",
        "shield_trigger": "S・トリガー",
        "evolution": "進化",
        "just_diver": "ジャストダイバー",
        "mach_fighter": "マッハファイター",
        "g_strike": "G・ストライク",
        "hyper_energy": "ハイパーエナジー",
        "shield_burn": "シールド焼却",
        "revolution_change": "革命チェンジ",
        "mekraid": "メクレイド",
        "friend_burst": "フレンド・バースト",
        "untap_in": "タップして出る",
        "meta_counter_play": "メタカウンター",
        "power_attacker": "パワーアタッカー",
        "g_zero": "G・ゼロ",
        "ex_life": "EXライフ",
        "mega_last_burst": "メガ・ラスト・バースト",
        "unblockable": "ブロックされない",
        "no_choice": "選ばれない",
        "attacker": "アタッカー",
        "s_trigger": "S・トリガー",
        
        # Uppercase versions (for compatibility)
        "SPEED_ATTACKER": "スピードアタッカー",
        "BLOCKER": "ブロッカー",
        "SLAYER": "スレイヤー",
        "DOUBLE_BREAKER": "W・ブレイカー",
        "TRIPLE_BREAKER": "T・ブレイカー",
        "WORLD_BREAKER": "ワールド・ブレイカー",
        "SHIELD_TRIGGER": "S・トリガー",
        "EVOLUTION": "進化",
        "JUST_DIVER": "ジャストダイバー",
        "MACH_FIGHTER": "マッハファイター",
        "G_STRIKE": "G・ストライク",
        "HYPER_ENERGY": "ハイパーエナジー",
        "SHIELD_BURN": "シールド焼却",
        "REVOLUTION_CHANGE": "革命チェンジ",
        "MEKRAID": "メクレイド",
        "FRIEND_BURST": "フレンド・バースト",
        "UNTAP_IN": "タップして出る",
        "META_COUNTER_PLAY": "メタカウンター",
        "POWER_ATTACKER": "パワーアタッカー",
        "G_ZERO": "G・ゼロ",
        "EX_LIFE": "EXライフ",
        "MEGA_LAST_BURST": "メガ・ラスト・バースト",
        "UNBLOCKABLE": "ブロックされない",
        "NO_CHOICE": "選ばれない",
        "ATTACKER": "アタッカー",
        "S_TRIGGER": "S・トリガー",
        "CANNOT_ATTACK": "攻撃できない",
        "CANNOT_BLOCK": "ブロックできない",
        "CANNOT_ATTACK_OR_BLOCK": "攻撃またはブロックできない",
        # New phrasing: cannot attack and cannot block (both)
        "CANNOT_ATTACK_AND_BLOCK": "攻撃もブロックもできない",
        "TARGET_RESTRICTION": "対象制限",
        "SPELL_RESTRICTION": "呪文制限",
        "TARGET_THIS_CANNOT_SELECT": "このクリーチャーを対象として選択できない",
        "TARGET_THIS_FORCE_SELECT": "可能ならこのクリーチャーを選択する"
    }

    # Duration Text Mapping (Added)
    DURATION_TRANSLATION: Dict[str, str] = {
        "PERMANENT": "常に",
        "THIS_TURN": "このターン",
        "UNTIL_END_OF_OPPONENT_TURN": "次の相手のターンの終わりまで",
        "UNTIL_START_OF_OPPONENT_TURN": "次の相手のターンのはじめまで",
        "UNTIL_END_OF_YOUR_TURN": "次の自分のターンの終わりまで",
        "UNTIL_START_OF_YOUR_TURN": "次の自分のターンのはじめまで",
        "DURING_OPPONENT_TURN": "次の相手のターン中"
    }
    
    # Zone Japanese mapping
    ZONE_JAPANESE: Dict[str, str] = {
        "HAND": "手札",
        "DECK": "山札",
        "DECK_TOP": "山札の上",
        "DECK_BOTTOM": "山札の下",
        "BATTLE_ZONE": "バトルゾーン",
        "MANA_ZONE": "マナゾーン",
        "SHIELD_ZONE": "シールドゾーン",
        "GRAVEYARD": "墓地",
        "BUFFER": "バッファ",
        "UNDER_CARD": "カードの下"
    }
    
    # Civilization Japanese mapping
    CIVILIZATION_JAPANESE: Dict[str, str] = {
        "LIGHT": "光",
        "WATER": "水",
        "DARKNESS": "闇",
        "FIRE": "火",
        "NATURE": "自然",
        "ZERO": "ゼロ"
    }
    
    # Card type Japanese mapping
    CARD_TYPE_JAPANESE: Dict[str, str] = {
        "CREATURE": "クリーチャー",
        "SPELL": "呪文",
        "CROSS_GEAR": "クロスギア",
        "CASTLE": "城",
        "EVOLUTION_CREATURE": "進化クリーチャー",
        "NEO_CREATURE": "NEOクリーチャー",
        "PSYCHIC_CREATURE": "サイキック・クリーチャー",
        "PSYCHIC_SUPER_CREATURE": "サイキック・スーパー・クリーチャー",
        "DRAGHEART_CREATURE": "ドラグハート・クリーチャー",
        "DRAGHEART_WEAPON": "ドラグハート・ウエポン",
        "DRAGHEART_FORTRESS": "ドラグハート・フォートレス",
        "AURA": "オーラ",
        "FIELD": "フィールド",
        "D2_FIELD": "D2フィールド"
    }

    PHASE_MAP: Dict[int, str] = {
        0: "ターン開始",
        1: "ドロー",
        2: "マナ",
        3: "メイン",
        4: "攻撃",
        5: "ブロック",
        6: "ターン終了"
    }

    ACTION_MAP: Dict[str, str] = {
        "DRAW_CARD": "カードを{value1}枚引く。",
        "ADD_MANA": "自分の山札の上から{value1}枚をマナゾーンに置く。",
        "DESTROY": "{target}を{value1}{unit}破壊する。",
        "TAP": "{target}を{value1}{unit}選び、タップする。",
        "UNTAP": "{target}を{value1}{unit}選び、アンタップする。",
        "RETURN_TO_HAND": "{target}を{value1}{unit}選び、手札に戻す。",
        "SEND_TO_MANA": "{target}を{value1}{unit}選び、マナゾーンに置く。",
        "MODIFY_POWER": "{target}のパワーを{value1}する。",
        "BREAK_SHIELD": "相手のシールドを{value1}つブレイクする。",
        "LOOK_AND_ADD": "自分の山札の上から{value1}枚を見る。その中から{value2}枚を手札に加え、残りを{zone}に置く。",
        "SEARCH_DECK_BOTTOM": "自分の山札の下から{value1}枚を見る。",
        "SEARCH_DECK": "自分の山札を見る。その中から{filter}を1枚選び、{zone}に置く。その後、山札をシャッフルする。",
        "MEKRAID": "メクレイド{value1}",
        "DISCARD": "手札を{value1}枚捨てる。",
        "PLAY_FROM_ZONE": "{source_zone}からコスト{value1}以下の{target}をプレイしてもよい。",
        "COUNT_CARDS": "{filter}の数を数える。",
        "GET_GAME_STAT": "（{str_val}を参照）",
        "REVEAL_CARDS": "山札の上から{value1}枚を表向きにする。",
        "SHUFFLE_DECK": "山札をシャッフルする。",
        "ADD_SHIELD": "山札の上から{value1}枚をシールド化する。",
        "SEND_SHIELD_TO_GRAVE": "相手のシールドを{value1}つ選び、墓地に置く。",
        "SEND_TO_DECK_BOTTOM": "{target}を{value1}枚、山札の下に置く。",
        "CAST_SPELL": "コストを支払わずに唱える。",
        "PUT_CREATURE": "{target}を{value1}{unit}バトルゾーンに出す。",
        "GRANT_KEYWORD": "{target}に「{str_val}」を与える。",
        "MOVE_CARD": "{target}を{zone}に置く。",
        "REPLACE_CARD_MOVE": "{target}を{from_zone}に置くかわりに{to_zone}に置く。",
        "COST_REFERENCE": "",
        "SUMMON_TOKEN": "「{str_val}」を{value1}体出す。",
        "RESET_INSTANCE": "{target}の状態をリセットする（アンタップ等）。",
        "REGISTER_DELAYED_EFFECT": "「{str_val}」の効果を{value1}ターン登録する。",
        "FRIEND_BURST": "{str_val}のフレンド・バースト",
        "MOVE_TO_UNDER_CARD": "{target}を{value1}{unit}選び、カードの下に置く。",
        "SELECT_NUMBER": "数字を1つ選ぶ。",
        "DECLARE_NUMBER": "{value1}～{value2}の数字を1つ宣言する。",
        "COST_REDUCTION": "{target}のコストを{value1}少なくする。ただし、コストは0以下にはならない。",
        "LOOK_TO_BUFFER": "{source_zone}から{value1}枚を見る（バッファへ）。",
        "SELECT_FROM_BUFFER": "バッファから{value1}枚選ぶ（{filter}）。",
        "PLAY_FROM_BUFFER": "バッファからプレイする。",
        "MOVE_BUFFER_TO_ZONE": "バッファから{zone}に置く。",
        "SELECT_OPTION": "次の中から選ぶ。",
        "LOCK_SPELL": "相手は呪文を唱えられない。",
        "REPLACE_CARD_MOVE": "{target}を{from_zone}に置くかわりに{to_zone}に置く。",
        "REPLACE_MOVE_CARD": "（置換移動）",
        "APPLY_MODIFIER": "効果を付与する。",
        
        # --- IF/IF_ELSE/ELSE Control Flow ---
        "IF": "（条件判定: {condition_detail}）",
        "IF_ELSE": "（条件分岐: {condition_detail}...）",
        "ELSE": "（そうでなければ）",

        # --- Generalized Commands (Mapped to natural text if encountered in Card Data) ---
        "TRANSITION": "{target}を{from_zone}から{to_zone}へ移動する。", # Fallback
        "MUTATE": "{target}の状態を変更する。", # Fallback
        "FLOW": "進行制御: {str_param}",
        "QUERY": "クエリ発行: {query_mode}",
        "ATTACH": "{target}を{base_target}の下に重ねる。",
        "GAME_RESULT": "ゲームを終了する（{result}）。",
    }

    STAT_KEY_MAP: Dict[str, Tuple[str, str]] = {
        "MANA_COUNT": ("マナゾーンのカード", "枚"),
        "CREATURE_COUNT": ("クリーチャー", "体"),
        "SHIELD_COUNT": ("シールド", "つ"),
        "HAND_COUNT": ("手札", "枚"),
        "GRAVEYARD_COUNT": ("墓地のカード", "枚"),
        "BATTLE_ZONE_COUNT": ("バトルゾーンのカード", "枚"),
        "OPPONENT_MANA_COUNT": ("相手のマナゾーンのカード", "枚"),
        "OPPONENT_CREATURE_COUNT": ("相手のクリーチャー", "体"),
        "OPPONENT_SHIELD_COUNT": ("相手のシールド", "つ"),
        "OPPONENT_HAND_COUNT": ("相手の手札", "枚"),
        "OPPONENT_GRAVEYARD_COUNT": ("相手の墓地のカード", "枚"),
        "OPPONENT_BATTLE_ZONE_COUNT": ("相手のバトルゾーンのカード", "枚"),
        "CARDS_DRAWN_THIS_TURN": ("このターンに引いたカード", "枚"),
        "MANA_CIVILIZATION_COUNT": ("マナゾーンの文明数", ""),
    }

    # Short aliases for natural language rendering of common zone transitions
    TRANSITION_ALIASES: Dict[Tuple[str, str], str] = {
        ("BATTLE_ZONE", "GRAVEYARD"): "破壊",
        ("HAND", "GRAVEYARD"): "捨てる",
        ("BATTLE_ZONE", "HAND"): "手札に戻す",
        ("DECK", "MANA_ZONE"): "マナチャージ",
        ("SHIELD_ZONE", "GRAVEYARD"): "シールド焼却",
        ("BATTLE_ZONE", "DECK"): "山札に戻す"
    }

    # Hint labels for how a linked input value is consumed by a subsequent step
    INPUT_USAGE_LABELS: Dict[str, str] = {
        "COST": "コストとして使用",
        "MAX_COST": "最大コストとして使用",
        "MIN_COST": "最小コストとして使用",
        "AMOUNT": "枚数として使用",
        "COUNT": "枚数として使用",
        "SELECTION": "選択数として使用",
        "TARGET_COUNT": "対象数として使用",
        "POWER": "パワーとして使用",
        "MAX_POWER": "最大パワーとして使用",
        "MIN_POWER": "最小パワーとして使用",
    }
    
    @classmethod
    def get_condition_text(cls, condition_type: str) -> str:
        """
        Get Japanese text for condition type.
        
        Args:
            condition_type: Condition type string (e.g., "DURING_YOUR_TURN")
        
        Returns:
            Japanese condition text, or original string if not found
        """
        return cls.CONDITION_JAPANESE.get(condition_type, condition_type)
    
    @classmethod
    def get_condition_type_label(cls, condition_type: str) -> str:
        """
        Get Japanese label for condition type (for GUI editor).
        
        Args:
            condition_type: Condition type string (e.g., "OPPONENT_DRAW_COUNT")
        
        Returns:
            Japanese condition type label, or original string if not found
        """
        return cls.CONDITION_TYPE_LABELS.get(condition_type, condition_type)
    
    @classmethod
    def get_scope_text(cls, scope: str) -> str:
        """
        Get Japanese text for scope/owner.
        
        Args:
            scope: Scope string (SELF, OPPONENT, ALL)
        
        Returns:
            Japanese scope prefix, or empty string if not found
        """
        return cls.SCOPE_JAPANESE.get(scope, "")
    
    @classmethod
    def get_trigger_text(cls, trigger: str, is_spell: bool = False) -> str:
        """
        Get Japanese text for trigger event.
        
        Args:
            trigger: Trigger type string
            is_spell: If True, use spell-specific trigger text where available
        
        Returns:
            Japanese trigger text
        """
        if is_spell and trigger in cls.SPELL_TRIGGER_JAPANESE:
            return cls.SPELL_TRIGGER_JAPANESE[trigger]
        return cls.TRIGGER_JAPANESE.get(trigger, trigger)
    
    @classmethod
    def get_modifier_type_text(cls, modifier_type: str) -> str:
        """
        Get Japanese text for modifier type.
        
        Args:
            modifier_type: Modifier type string
        
        Returns:
            Japanese modifier type text
        """
        return cls.MODIFIER_TYPE_JAPANESE.get(modifier_type, modifier_type)
    
    @classmethod
    def get_keyword_text(cls, keyword: str) -> str:
        """
        Get Japanese text for keyword.
        
        Args:
            keyword: Keyword string (case-insensitive lookup)
        
        Returns:
            Japanese keyword text, or original string if not found
        """
        # Try exact match first
        if keyword in cls.KEYWORD_TRANSLATION:
            return cls.KEYWORD_TRANSLATION[keyword]
        # Try lowercase match
        if keyword.lower() in cls.KEYWORD_TRANSLATION:
            return cls.KEYWORD_TRANSLATION[keyword.lower()]
        # Try uppercase match
        if keyword.upper() in cls.KEYWORD_TRANSLATION:
            return cls.KEYWORD_TRANSLATION[keyword.upper()]
        return keyword
    
    @classmethod
    def get_zone_text(cls, zone: str) -> str:
        """
        Get Japanese text for zone.
        
        Args:
            zone: Zone string
        
        Returns:
            Japanese zone text
        """
        return cls.ZONE_JAPANESE.get(zone, zone)
    
    @classmethod
    def get_civilization_text(cls, civ: str) -> str:
        """
        Get Japanese text for civilization.
        
        Args:
            civ: Civilization string
        
        Returns:
            Japanese civilization text
        """
        return cls.CIVILIZATION_JAPANESE.get(civ, civ)
    
    @classmethod
    def get_card_type_text(cls, card_type: str) -> str:
        """
        Get Japanese text for card type.
        
        Args:
            card_type: Card type string
        
        Returns:
            Japanese card type text
        """
        return cls.CARD_TYPE_JAPANESE.get(card_type, card_type)

    @classmethod
    def get_duration_text(cls, duration_key: str) -> str:
        """
        Get Japanese text for duration.
        """
        return cls.DURATION_TRANSLATION.get(duration_key, duration_key)
    
    @classmethod
    def get_stat_key_label(cls, stat_key: str) -> str:
        """
        Get Japanese label for stat key (for GUI editor).
        
        Args:
            stat_key: Stat key string (e.g., "MY_SHIELD_COUNT")
        
        Returns:
            Japanese stat key label with unit
        """
        if stat_key in cls.STAT_KEY_MAP:
            name, unit = cls.STAT_KEY_MAP[stat_key]
            return f"{name}（{unit}）" if unit else name
        return stat_key
