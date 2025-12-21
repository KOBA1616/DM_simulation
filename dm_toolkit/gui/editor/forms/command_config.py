# -*- coding: utf-8 -*-
from dm_toolkit.gui.localization import tr

COMMAND_UI_CONFIG = {
    "NONE": {"visible": []},

    # --- Atomic Actions (Shortcuts) ---
    "DRAW_CARD": {
        "visible": ["target_group", "amount", "optional", "input_link"],
        "label_amount": "Cards to Draw",
        "produces_output": True,
        "outputs": {"output_value_key": "Cards Drawn"}
    },
    "DISCARD": {
        "visible": ["target_group", "target_filter", "amount", "optional", "input_link", "output_link"],
        "label_amount": "Count",
        "produces_output": True,
        "outputs": {"output_value_key": "Discarded Cards"}
    },
    "DESTROY": {
        "visible": ["target_group", "target_filter", "amount", "input_link"],
        "label_amount": "Count (if selecting)",
    },
    "MANA_CHARGE": {
        "visible": ["target_group", "target_filter", "amount", "input_link"],
        "label_amount": "Count (if selecting)",
    },
    "TAP": {
        "visible": ["target_group", "target_filter", "input_link"],
    },
    "UNTAP": {
        "visible": ["target_group", "target_filter", "input_link"],
    },
    "RETURN_TO_HAND": {
        "visible": ["target_group", "target_filter", "amount", "input_link"],
        "label_amount": "Count (if selecting)",
    },
    "BREAK_SHIELD": {
        "visible": ["target_group", "target_filter", "amount", "input_link"],
        "label_amount": "Count",
    },

    # --- Generalized Commands ---
    "TRANSITION": {
        "visible": ["target_group", "target_filter", "from_zone", "to_zone", "amount", "optional", "input_link", "output_link"],
        "label_amount": "Count",
        "produces_output": True,
        "outputs": {"output_value_key": "Moved Cards"}
    },
    "MUTATE": {
        "visible": ["target_group", "target_filter", "mutation_kind", "amount", "str_param"],
        "label_mutation_kind": "Mutation Type",
        "label_amount": "Value / Duration",
        "label_str_param": "Extra Param"
    },
    "POWER_MOD": {
        "visible": ["target_group", "target_filter", "amount"],
        "label_amount": "Power Adjustment"
    },
    "ADD_KEYWORD": {
        "visible": ["target_group", "target_filter", "mutation_kind", "amount"],
        "label_mutation_kind": "Keyword",
        "label_amount": "Duration (Turns)" # Assuming amount maps to duration in engine for keywords?
    },

    # --- Information / Flow ---
    "QUERY": {
        "visible": ["target_group", "target_filter", "query_mode", "output_link"],
        # Query mode logic handles internal visibility of filter
        "produces_output": True,
        "outputs": {"output_value_key": "Query Result"}
    },
    "FLOW": {
        "visible": ["str_param"],
        "label_str_param": "Flow Instruction"
    },
    "SEARCH_DECK": {
        "visible": ["target_filter", "amount", "output_link"],
        "label_amount": "Count",
        "produces_output": True,
        "outputs": {"output_value_key": "Found Cards"}
    },
    "SHIELD_TRIGGER": {
        "visible": ["target_group"], # Usually implicit?
    }
}
