# -*- coding: utf-8 -*-
"""Shared stat-key registry used by query/condition/editor layers.

再発防止: query_mode と stat_key を別ファイルで重複管理すると、
追加時に UI 候補・テキスト表示・実行系の同期漏れが起きやすい。
このモジュールを単一ソースとして各層から参照する。
"""

from typing import Tuple


# QueryCommand(str_param) で統計値を取得できる stat keys
QUERY_STAT_KEYS: Tuple[str, ...] = (
    "MANA_COUNT",
    "CREATURE_COUNT",
    "SHIELD_COUNT",
    "HAND_COUNT",
    "GRAVEYARD_COUNT",
    "BATTLE_ZONE_COUNT",
    "OPPONENT_MANA_COUNT",
    "OPPONENT_CREATURE_COUNT",
    "OPPONENT_SHIELD_COUNT",
    "OPPONENT_HAND_COUNT",
    "OPPONENT_GRAVEYARD_COUNT",
    "OPPONENT_BATTLE_ZONE_COUNT",
    "CARDS_DRAWN_THIS_TURN",
    "SPELL_CAST_THIS_TURN",
    "MANA_CIVILIZATION_COUNT",
)


# ConditionWidget の COMPARE_STAT 候補
COMPARE_STAT_EDITOR_KEYS: Tuple[str, ...] = (
    "MY_MANA_COUNT",
    "OPPONENT_MANA_COUNT",
    "MY_HAND_COUNT",
    "OPPONENT_HAND_COUNT",
    "MY_SHIELD_COUNT",
    "OPPONENT_SHIELD_COUNT",
    "MY_BATTLE_ZONE_COUNT",
    "OPPONENT_BATTLE_ZONE_COUNT",
    "SUMMON_COUNT_THIS_TURN",
    "CARDS_DRAWN_THIS_TURN",
    "SPELL_CAST_THIS_TURN",
    "DESTROY_COUNT_THIS_TURN",
    "MANA_SET_THIS_TURN",
    "SHIELD_BREAK_ATTEMPT_THIS_TURN",
    "SHIELD_BREAK_RESOLVED_THIS_TURN",
    "ATTACKED_THIS_TURN",
    "MY_ATTACKED_THIS_TURN",
    "OPPONENT_ATTACKED_THIS_TURN",
)


# Query/Condition UI でよく使う統計キー
EDITOR_QUICK_STATS_KEYS: Tuple[str, ...] = (
    "MANA_CIVILIZATION_COUNT",
    "SHIELD_COUNT",
    "HAND_COUNT",
    "CARDS_DRAWN_THIS_TURN",
)
