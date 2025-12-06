
# Mapping configuration for Action Form UI
# Format: ActionType -> {
#   val1_label: str (key for tr), val1_visible: bool,
#   val2_label: str, val2_visible: bool,
#   str_label: str, str_visible: bool,
#   filter_visible: bool,
#   tooltip: str,
#   produces_output: bool # New flag for Automated Variable Linking
# }

ACTION_UI_CONFIG = {
    "DRAW_CARD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Draws [Value 1] cards from the deck.",
        "produces_output": True # Often outputs number drawn
    },
    "ADD_MANA": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Puts [Value 1] cards from the top of the deck into the mana zone.",
        "produces_output": True # Outputs number added
    },
    "DESTROY": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Destroys [Value 1] cards matching the Filter.",
        "produces_output": True # Outputs number destroyed
    },
    "RETURN_TO_HAND": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Returns [Value 1] cards matching the Filter to hand.",
        "produces_output": True
    },
    "SEARCH_DECK_BOTTOM": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Searches deck for [Value 1] cards matching Filter, reveals them, and puts them at the bottom.",
        "produces_output": True
    },
    "MEKRAID": {
        "val1_label": "Cost (Level)", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Race (Optional)", "str_visible": True,
        "filter_visible": True,
        "tooltip": "Mekraid [Level]. Filter/String can specify Race.",
        "produces_output": True # Success/Fail or Cost?
    },
    "TAP": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Taps [Value 1] cards matching Filter.",
        "produces_output": True
    },
    "UNTAP": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Untaps [Value 1] cards matching Filter.",
        "produces_output": True
    },
    "COST_REFERENCE": {
        "val1_label": "Reduction Amount", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Keyword (e.g. HYPER_ENERGY)", "str_visible": True,
        "filter_visible": True,
        "tooltip": "Reduces cost by [Value 1]. Filter defines required cards (e.g. for Sympathy).",
        "produces_output": True
    },
    "BREAK_SHIELD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Breaks [Value 1] shields.",
        "produces_output": True
    },
    "LOOK_AND_ADD": {
        "val1_label": "Look Count", "val1_visible": True,
        "val2_label": "Add Count", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Look at top [Value 1] cards, add [Value 2] matching Filter to hand, rest to bottom.",
        "produces_output": True
    },
    "SUMMON_TOKEN": {
        "val1_label": "Token ID", "val1_visible": True,
        "val2_label": "Count", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Summons [Value 2] tokens of ID [Value 1].",
        "produces_output": True
    },
    "DISCARD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Discards [Value 1] cards matching Filter.",
        "produces_output": True
    },
    "REVOLUTION_CHANGE": {
        "val1_label": "Cost", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Civilization", "str_visible": True,
        "filter_visible": True,
        "tooltip": "Revolution Change logic.",
        "produces_output": True
    },
    "COUNT_CARDS": {
        "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Mode (CARDS/CIVILIZATIONS)", "str_visible": True,
        "filter_visible": True,
        "tooltip": "Counts cards or civilizations matching Filter and stores in Output Key.",
        "produces_output": True
    },
    "APPLY_MODIFIER": {
        "val1_label": "Power Mod / Value", "val1_visible": True,
        "val2_label": "Duration (Turns)", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Applies modification (e.g. Power +[Value 1]) to Filtered cards for [Value 2] turns.",
        "produces_output": False # Usually doesn't return a value to link
    },
    "REVEAL_CARDS": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Reveals [Value 1] cards matching Filter.",
        "produces_output": True
    },
    "SEND_TO_DECK_BOTTOM": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Sends [Value 1] cards matching Filter to deck bottom.",
        "produces_output": True
    },
    "PLAY_FROM_ZONE": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Plays cards from zone.",
        "produces_output": True
    },
    "REGISTER_DELAYED_EFFECT": {
         "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Registers a delayed effect.",
        "produces_output": False
    },
    "RESET_INSTANCE": {
         "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Resets card instance.",
        "produces_output": False
    },
    "SELECT_NUMBER": {
        "val1_label": "Max Value (Static)", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Prompt AI to select a number between Min and Max. Use 'Input Key' to set Max dynamically.",
        "produces_output": True
    },
     "NONE": {
        "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "No action.",
        "produces_output": False
    }
}
