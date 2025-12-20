# -*- coding: cp932 -*-
from dm_toolkit.gui.localization import tr

ACTION_UI_CONFIG = {
    "NONE": {"visible": []},
    "DRAW_CARD": {
        "visible": ["value1", "input_value_key"],
        "label_value1": "Cards to Draw",
        "can_be_optional": True,
        "produces_output": True,
        "outputs": {"output_value_key": "Cards Drawn"},
        "inputs": {"input_value_key": "Count Override (optional)"}
    },
    "ADD_MANA": {
        "visible": ["value1"],
        "label_value1": "Cards to Charge",
        "can_be_optional": True
    },
    "MODIFY_POWER": {
        "visible": ["scope", "filter", "value1", "value2"],
        "label_value1": "Power Mod",
        "label_value2": "Duration (Turns)",
        "can_be_optional": True
    },
    "BREAK_SHIELD": {
        "visible": ["value1", "input_value_key"],
        "label_value1": "Shields to Break",
        "inputs": {"input_value_key": "Count Override (optional)"}
    },
    "LOOK_AND_ADD": {
        "visible": ["value1", "value2", "filter", "output_value_key"],
        "label_value1": "Look at N",
        "label_value2": "Add M to Hand",
        "produces_output": True,
        "outputs": {"output_value_key": "Added Cards"},
    },
    "SUMMON_TOKEN": {
        "visible": ["value1", "str_val"],
        "label_value1": "Count",
        "label_str_val": "Token ID/Name"
    },
    "SEARCH_DECK": {
        "visible": ["filter", "output_value_key", "input_value_key"],
        "produces_output": True,
        "allowed_filter_fields": ["civilizations", "races", "types", "cost", "power"], # Mask zones
        "outputs": {"output_value_key": "Selected Cards"},
        "inputs": {"input_value_key": "Count Override (optional)"}
    },
    "MEKRAID": {
        "visible": ["value1", "filter"],
        "label_value1": "Level"
    },
    "PLAY_FROM_ZONE": {
        "visible": ["source_zone", "filter", "value1"],
        "label_value1": "Cost Reduction"
    },
    "COST_REFERENCE": {
        "visible": ["value1", "str_val"], # str_val as ComboBox for Ref Mode
        "label_value1": "Multiplier"
    },
    "COST_REDUCTION": { # New dedicated UI type for generic reductions
        "visible": ["value1", "filter"],
        "label_value1": "Reduction Amount"
    },
    "LOOK_TO_BUFFER": {
        "visible": ["value1", "source_zone", "input_value_key"],
        "label_value1": "Count",
        "inputs": {"input_value_key": "Count Override"}
    },
    "SELECT_FROM_BUFFER": {
        "visible": ["filter", "value1", "output_value_key", "input_value_key"],
        "label_value1": "Count",
        "produces_output": True,
        "outputs": {"output_value_key": "Selected Cards"},
        "inputs": {"input_value_key": "Count Override"}
    },
    "PLAY_FROM_BUFFER": {
        "visible": ["filter"]
    },
    "MOVE_BUFFER_TO_ZONE": {
        "visible": ["destination_zone"]
    },
    "REVOLUTION_CHANGE": {
        "visible": ["filter"], # Filter for valid attackers
        "allowed_filter_fields": ["civilizations", "races", "cost"] # Mask zones (implied Battle Zone)
    },
    "COUNT_CARDS": {
        "visible": ["scope", "filter", "output_value_key", "str_val"], # str_val for Mode
        "label_str_val": "Mode (Optional)",
        "produces_output": True,
        "outputs": {"output_value_key": "Count Result"}
    },
    "GET_GAME_STAT": {
        "visible": ["str_val", "output_value_key"],
        "label_str_val": "Stat Name",
        "produces_output": True,
        "outputs": {"output_value_key": "Stat Value"}
    },
    "APPLY_MODIFIER": {
        "visible": ["filter", "str_val", "value1", "value2"],
        "label_str_val": "Modifier Type",
        "label_value1": "Value",
        "label_value2": "Duration (Turns)"
    },
    "REVEAL_CARDS": {
        "visible": ["scope", "filter", "value1", "input_value_key"],
        "label_value1": "Count (from Top)",
        "inputs": {"input_value_key": "Count Override"}
    },
    "REGISTER_DELAYED_EFFECT": {
        "visible": ["str_val", "value1"],
        "label_str_val": "Effect ID/Name",
        "label_value1": "Duration"
    },
    "RESET_INSTANCE": {
        "visible": ["scope", "filter"]
    },
    "SHUFFLE_DECK": {
        "visible": ["scope"] # Usually Self or Target Player
    },
    "ADD_SHIELD": {
        "visible": ["scope", "value1", "input_value_key"],
        "label_value1": "Count",
        "inputs": {"input_value_key": "Count Override"}
    },
    "MOVE_TO_UNDER_CARD": {
        "visible": ["scope", "filter", "value1", "input_value_key"],
        "label_value1": "Count",
        "inputs": {"input_value_key": "Target Card (Destination)"}
    },
    "SELECT_NUMBER": {
        "visible": ["output_value_key", "value1"],
        "label_value1": "Default/Max (Heuristic)",
        "produces_output": True,
        "outputs": {"output_value_key": "Selected Number"}
    },
    "DECLARE_NUMBER": {
        "visible": ["output_value_key", "value1", "value2"],
        "label_value1": "Min Value",
        "label_value2": "Max Value",
        "produces_output": True,
        "outputs": {"output_value_key": "Declared Number"}
    },
    "FRIEND_BURST": {
        "visible": ["str_val", "filter"],
        "label_str_val": "Race (e.g. Fire Bird)",
        "can_be_optional": True
    },
    "GRANT_KEYWORD": {
        "visible": ["scope", "filter", "str_val", "value2"],
        "label_str_val": "Keyword",
        "label_value2": "Duration (Turns)"
    },
    "MOVE_CARD": {
        "visible": ["scope", "filter", "destination_zone", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "SELECT_OPTION": {
        "visible": ["value1", "value2", "str_val"],
        "label_value1": "Select Count",
        "label_value2": "Allow Duplicates",
        "label_str_val": "Mode Name (Optional)",
        "can_be_optional": True
    },
    # Legacy / Consolidated Actions (Hidden from list but kept for compatibility config)
    "DESTROY": {
        "visible": ["scope", "filter", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "TAP": {
        "visible": ["scope", "filter", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "UNTAP": {
        "visible": ["scope", "filter", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "RETURN_TO_HAND": {
        "visible": ["scope", "filter", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "SEND_TO_MANA": {
        "visible": ["scope", "filter", "target_choice", "input_value_key"],
        "can_be_optional": True,
        "inputs": {"input_value_key": "Target Selection (Pre-selected)"}
    },
    "SEARCH_DECK_BOTTOM": {
        "visible": ["value1", "filter", "output_value_key", "input_value_key"],
        "label_value1": "Reveal N (from Bottom)",
        "produces_output": True,
        "outputs": {"output_value_key": "Selected Cards"},
        "inputs": {"input_value_key": "Count Override"}
    },
    "DISCARD": {
        "visible": ["scope", "filter", "value1", "target_choice", "output_value_key", "input_value_key"],
        "label_value1": "Count",
        "can_be_optional": True,
        "produces_output": True,
        "outputs": {"output_value_key": "Discarded Cards"},
        "inputs": {"input_value_key": "Count Override"}
    },
    "SEND_SHIELD_TO_GRAVE": {
        "visible": ["scope", "filter", "value1", "input_value_key"],
        "label_value1": "Count",
        "inputs": {"input_value_key": "Count Override"}
    },
    "SEND_TO_DECK_BOTTOM": {
        "visible": ["scope", "filter", "value1", "input_value_key"],
        "label_value1": "Count",
        "inputs": {"input_value_key": "Count Override"}
    }
}
