# -*- coding: utf-8 -*-
"""
Centralized text resources for Static Abilities and Trigger Effects.
Maps trigger types, conditions, scopes, and modifier types to Japanese text.
"""

from typing import Dict, Optional


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
    
    # Scope/Owner Japanese mapping (STATIC ability context)
    SCOPE_JAPANESE: Dict[str, str] = {
        "SELF": "自分の",
        "OPPONENT": "相手の",
        "ALL": ""
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
        "SET_KEYWORD": "キーワード設定"
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
        "CANNOT_ATTACK": "攻撃できない",
        "CANNOT_BLOCK": "ブロックできない",
        "CANNOT_ATTACK_OR_BLOCK": "攻撃またはブロックできない"
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
