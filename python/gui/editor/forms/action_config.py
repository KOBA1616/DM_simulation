
# Mapping configuration for Action Form UI
# Format: ActionType -> {
#   val1_label: str (key for tr), val1_visible: bool,
#   val2_label: str, val2_visible: bool,
#   str_label: str, str_visible: bool,
#   filter_visible: bool,
#   tooltip: str
# }

ACTION_UI_CONFIG = {
    "DRAW_CARD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Draws [Value 1] cards from the deck."
    },
    "ADD_MANA": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False, # Usually top of deck
        "tooltip": "Puts [Value 1] cards from the top of the deck into the mana zone."
    },
    "DESTROY": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Destroys [Value 1] cards matching the Filter."
    },
    "RETURN_TO_HAND": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Returns [Value 1] cards matching the Filter to hand."
    },
    "SEARCH_DECK_BOTTOM": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Searches deck for [Value 1] cards matching Filter, reveals them, and puts them at the bottom."
    },
    "MEKRAID": {
        "val1_label": "Cost (Level)", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Race (Optional)", "str_visible": True,
        "filter_visible": True, # For specific Mekraid conditions? Or just civilization in str? Actually Mekraid is Race+Level usually.
        # Implementation details of Mekraid usually take Civ from card, Level from val1, Race from str_val
        "tooltip": "Mekraid [Level]. Filter/String can specify Race."
    },
    "TAP": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Taps [Value 1] cards matching Filter."
    },
    "UNTAP": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Untaps [Value 1] cards matching Filter."
    },
    "COST_REFERENCE": {
        "val1_label": "Reduction Amount", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Keyword (e.g. HYPER_ENERGY)", "str_visible": True,
        "filter_visible": True, # Filter defines what to tap/count
        "tooltip": "Reduces cost by [Value 1]. Filter defines required cards (e.g. for Sympathy)."
    },
    "BREAK_SHIELD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Breaks [Value 1] shields."
    },
    "LOOK_AND_ADD": {
        "val1_label": "Look Count", "val1_visible": True,
        "val2_label": "Add Count", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Look at top [Value 1] cards, add [Value 2] matching Filter to hand, rest to bottom."
    },
    "SUMMON_TOKEN": {
        "val1_label": "Token ID", "val1_visible": True, # Actually requires Card ID?
        "val2_label": "Count", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "Summons [Value 2] tokens of ID [Value 1]."
    },
    "DISCARD": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True, # Random or Selected?
        "tooltip": "Discards [Value 1] cards matching Filter."
    },
    "REVOLUTION_CHANGE": {
        "val1_label": "Cost", "val1_visible": True, # Usually Cost >= X
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Civilization", "str_visible": True,
        "filter_visible": True, # Defines the condition (e.g. Race Dragon)
        "tooltip": "Revolution Change logic."
    },
    "COUNT_CARDS": {
        "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True, # Essential
        "tooltip": "Counts cards matching Filter and stores in Output Key."
    },
    "GET_GAME_STAT": {
        "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "Stat Type (e.g. MANA_CIVILIZATIONS)", "str_visible": True,
        "filter_visible": False,
        "tooltip": "Gets a game statistic and stores in Output Key."
    },
    "APPLY_MODIFIER": {
        "val1_label": "Power Mod / Value", "val1_visible": True,
        "val2_label": "Duration (Turns)", "val2_visible": True,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True, # Target
        "tooltip": "Applies modification (e.g. Power +[Value 1]) to Filtered cards for [Value 2] turns."
    },
    "REVEAL_CARDS": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Reveals [Value 1] cards matching Filter."
    },
    "SEND_TO_DECK_BOTTOM": {
        "val1_label": "Count", "val1_visible": True,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": True,
        "tooltip": "Sends [Value 1] cards matching Filter to deck bottom."
    },
     "NONE": {
        "val1_label": "Value 1", "val1_visible": False,
        "val2_label": "Value 2", "val2_visible": False,
        "str_label": "String Value", "str_visible": False,
        "filter_visible": False,
        "tooltip": "No action."
    }
}
