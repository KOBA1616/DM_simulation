# -*- coding: cp932 -*-
# Configuration for Command Edit Form UI

# Defines which fields are visible for each CommandType.
# Fields:
# - target_group
# - target_filter
# - amount
# - str_param
# - optional
# - from_zone
# - to_zone
# - mutation_kind

COMMAND_UI_CONFIG = {
    "NONE": {
        "visible": []
    },
    "TRANSITION": {
        "visible": ["target_group", "target_filter", "to_zone", "optional"],
        "tooltip": "Move cards between zones"
    },
    "MUTATE": {
        "visible": ["target_group", "target_filter", "mutation_kind", "amount", "str_param"],
        "tooltip": "Modify card properties (Power, Tapped, etc)"
    },
    "FLOW": {
        "visible": ["condition"], # Branching is handled by tree structure
        "tooltip": "Control flow (If/Else)"
    },
    "QUERY": {
        "visible": ["target_group", "target_filter", "amount"], # Amount as count required?
        "tooltip": "Select targets or count cards"
    },
    "DRAW_CARD": {
        "visible": ["amount", "optional"],
        "tooltip": "Draw cards from deck"
    },
    "DISCARD": {
        "visible": ["target_group", "target_filter", "amount", "optional"], # Random or Select
        "tooltip": "Discard cards from hand"
    },
    "DESTROY": {
        "visible": ["target_group", "target_filter", "amount", "optional"],
        "tooltip": "Destroy creatures"
    },
    "MANA_CHARGE": {
        "visible": ["target_group", "target_filter", "from_zone", "amount", "optional"],
        "tooltip": "Put cards into mana zone"
    },
    "TAP": {
        "visible": ["target_group", "target_filter", "amount", "optional"],
        "tooltip": "Tap cards"
    },
    "UNTAP": {
        "visible": ["target_group", "target_filter", "amount", "optional"],
        "tooltip": "Untap cards"
    },
    "POWER_MOD": {
        "visible": ["target_group", "target_filter", "amount"],
        "tooltip": "Modify power of creatures"
    },
    "ADD_KEYWORD": {
        "visible": ["target_group", "target_filter", "mutation_kind"], # mutation_kind as Keyword
        "tooltip": "Grant keyword abilities"
    },
    "RETURN_TO_HAND": {
        "visible": ["target_group", "target_filter", "amount", "optional"],
        "tooltip": "Return cards to hand"
    },
    "BREAK_SHIELD": {
        "visible": ["target_group", "target_filter", "amount"],
        "tooltip": "Break shields"
    },
    "SEARCH_DECK": {
        "visible": ["target_filter", "amount", "to_zone", "optional"],
        "tooltip": "Search deck for cards"
    },
    "SHIELD_TRIGGER": {
        "visible": ["target_filter"], # Usually implies self
        "tooltip": "Activate Shield Trigger"
    }
}
