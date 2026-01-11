# -*- coding: utf-8 -*-
# Condition UI Configuration
from dm_toolkit.gui.localization import tr

CONDITION_UI_CONFIG = {
    "NONE": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "MANA_ARMED": {
        "show_val": True,
        "show_str": True, # Usually specifies civ
        "label_val": "Count",
        "label_str": "Civilization"
    },
    "SHIELD_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count",
        "label_str": "Comparison (Optional)"
    },
    "CIVILIZATION_MATCH": {
        "show_val": False,
        "show_str": True,
        "label_val": "Value",
        "label_str": "Civilization"
    },
    "OPPONENT_PLAYED_WITHOUT_MANA": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "OPPONENT_DRAW_COUNT": {
        "show_val": True,
        "show_str": False,
        "label_val": "Count (>=)",
        "label_str": "String"
    },
    "DURING_YOUR_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "DURING_OPPONENT_TURN": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "FIRST_ATTACK": {
        "show_val": False,
        "show_str": False,
        "label_val": "Value",
        "label_str": "String"
    },
    "EVENT_FILTER_MATCH": {
        "show_val": False,
        "show_str": False,
        "show_filter": True
    }
}
