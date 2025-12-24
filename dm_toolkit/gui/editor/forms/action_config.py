# -*- coding: utf-8 -*-
from dm_toolkit.gui.localization import tr
try:
    import dm_ai_module as m
except ImportError:
    m = None

# Define Categories and Sub-Actions (The "Menu" Structure)
ACTION_CATEGORIES = {
    "ZONE_CONTROL": {
        "label": "Zone Control (Move)",
        "actions": [
            "DRAW_CARD",
            "DESTROY",
            "SEND_TO_MANA",
            "BOUNCE",
            "SHIELD_BREAK",
            "ADD_SHIELD_TO_HAND",
            "ADD_SHIELD",
            "DISCARD",
            "PUT_INTO_BATTLE",
            "SEND_TO_DECK"
        ]
    },
    "STATE_MUTATION": {
        "label": "State & Mutation",
        "actions": [
            "MODIFY_POWER",
            "GRANT_ABILITY",
            "TAP",
            "UNTAP",
            "SHIELD_BURN",
            "COST_REDUCTION"
        ]
    },
    "SEARCH_REVEAL": {
        "label": "Search & Reveal",
        "actions": [
            "SEARCH_DECK",
            "LOOK_AND_ADD",
            "REVEAL_CARDS",
            "SHUFFLE_DECK"
        ]
    },
    "MULTILAYER": {
        "label": "Multilayer Card Operations",
        "actions": [
            "REVOLUTION_CHANGE",
            "MEKRAID",
            "MOVE_UNDER",
            "RESET_INSTANCE"
        ]
    },
    "LOGIC_SPECIAL": {
        "label": "Logic & Special",
        "actions": [
            "SUMMON_TOKEN",
            "COUNT_CARDS",
            "GET_GAME_STAT",
            "SELECT_NUMBER",
            "SELECT_OPTION",
            "FRIEND_BURST",
            "CAST_SPELL"
        ]
    }
}

# Define UI Configuration for each Sub-Action
# This maps the UI selection to Visible Fields and Underlying Data
ACTION_UI_CONFIG = {
    # --- ZONE CONTROL ---
    "DRAW_CARD": {
        "label": "Draw Card",
        "visible": ["value1"],
        "label_value1": "Count",
        "produces_output": True,
        "outputs": {"output_value_key": "Cards Drawn"},
        "json_template": {"type": "MOVE_CARD", "from_zone": "DECK", "to_zone": "HAND"}
    },
    "DESTROY": {
        "label": "Destroy",
        "visible": ["scope", "filter", "target_choice"],
        "can_be_optional": True,
        "json_template": {"type": "MOVE_CARD", "to_zone": "GRAVEYARD", "from_zone": "BATTLE_ZONE"}
    },
    "SEND_TO_MANA": {
        "label": "Send to Mana",
        "visible": ["scope", "filter", "target_choice", "source_zone"], # Source zone configurable? Usually Battle or Hand.
        "can_be_optional": True,
        "json_template": {"type": "MOVE_CARD", "to_zone": "MANA_ZONE"}
    },
    "BOUNCE": {
        "label": "Return to Hand (Bounce)",
        "visible": ["scope", "filter", "target_choice", "source_zone"],
        "can_be_optional": True,
        "json_template": {"type": "MOVE_CARD", "to_zone": "HAND"}
    },
    "SHIELD_BREAK": {
        "label": "Shield Break (Triggers OK)",
        "visible": ["scope", "filter", "target_choice", "value1"], # Value1 for count if selecting multiple? Usually filter handles count.
        "label_value1": "Count (if random/top)",
        "json_template": {"type": "MOVE_CARD", "from_zone": "SHIELD_ZONE", "to_zone": "HAND", "str_val": "BREAK"}
    },
    "ADD_SHIELD_TO_HAND": {
        "label": "Add Shield to Hand (No Triggers)",
        "visible": ["scope", "filter", "target_choice", "value1"],
        "label_value1": "Count",
        "json_template": {"type": "MOVE_CARD", "from_zone": "SHIELD_ZONE", "to_zone": "HAND", "str_val": "RETURN"}
    },
    "ADD_SHIELD": {
        "label": "Add Shield",
        "visible": ["scope", "filter", "value1", "source_zone"], # Source can be Deck or Hand
        "label_value1": "Count",
        "json_template": {"type": "ADD_SHIELD"} # Keep primitive or wrap in MOVE_CARD to SHIELD_ZONE? MOVE_CARD is better if source is flexible.
        # But ADD_SHIELD primitive handles "Top of Deck" implicitly if source not specified.
        # Let's map to ADD_SHIELD primitive for now to be safe, or MOVE_CARD if source specified.
    },
    "DISCARD": {
        "label": "Discard",
        "visible": ["scope", "filter", "target_choice", "value1"],
        "label_value1": "Count (Random/Select)",
        "produces_output": True,
        "json_template": {"type": "MOVE_CARD", "from_zone": "HAND", "to_zone": "GRAVEYARD"}
    },
    "PUT_INTO_BATTLE": {
        "label": "Put into Battle",
        "visible": ["source_zone", "filter", "scope"],
        "json_template": {"type": "MOVE_CARD", "to_zone": "BATTLE_ZONE"}
    },
    "SEND_TO_DECK": {
        "label": "Send to Deck",
        "visible": ["scope", "filter", "target_choice", "str_val"],
        "label_str_val": "Position (TOP/BOTTOM)",
        "json_template": {"type": "MOVE_CARD", "to_zone": "DECK"}
    },

    # --- STATE & MUTATION ---
    "MODIFY_POWER": {
        "label": "Modify Power",
        "visible": ["scope", "filter", "value1", "value2"],
        "label_value1": "Power Mod",
        "label_value2": "Duration (Turns)",
        "json_template": {"type": "MODIFY_POWER"} # Or APPLY_MODIFIER generic
    },
    "GRANT_ABILITY": {
        "label": "Grant Ability",
        "visible": ["scope", "filter", "str_val", "value2"],
        "label_str_val": "Keyword (e.g. blocker)",
        "label_value2": "Duration (Turns)",
        "json_template": {"type": "GRANT_KEYWORD"}
    },
    "TAP": {
        "label": "Tap",
        "visible": ["scope", "filter", "target_choice"],
        "json_template": {"type": "MUTATE", "str_val": "TAP"}
    },
    "UNTAP": {
        "label": "Untap",
        "visible": ["scope", "filter", "target_choice"],
        "json_template": {"type": "MUTATE", "str_val": "UNTAP"}
    },
    "SHIELD_BURN": {
        "label": "Shield Burn",
        "visible": ["scope", "filter", "value1"],
        "label_value1": "Count",
        "json_template": {"type": "MUTATE", "str_val": "SHIELD_BURN"}
    },
    "COST_REDUCTION": {
        "label": "Cost Reduction",
        "visible": ["filter", "value1", "source_zone"], # Filter defines what cards are reduced
        "label_value1": "Reduction Amount",
        "json_template": {"type": "APPLY_MODIFIER", "str_val": "COST"}
    },

    # --- SEARCH & REVEAL ---
    "SEARCH_DECK": {
        "label": "Search Deck",
        "visible": ["filter", "output_value_key", "value1"], # Value1 for max count
        "produces_output": True,
        "json_template": {"type": "SEARCH_DECK"}
    },
    "LOOK_AND_ADD": {
        "label": "Look & Add",
        "visible": ["value1", "value2", "filter", "output_value_key"],
        "label_value1": "Look N",
        "label_value2": "Add M",
        "produces_output": True,
        "json_template": {"type": "LOOK_AND_ADD"}
    },
    "REVEAL_CARDS": {
        "label": "Reveal Cards",
        "visible": ["scope", "filter", "value1"],
        "json_template": {"type": "REVEAL_CARDS"}
    },
    "SHUFFLE_DECK": {
        "label": "Shuffle Deck",
        "visible": ["scope"], # Usually owner of deck
        "json_template": {"type": "SHUFFLE_DECK"}
    },

    # --- MULTILAYER ---
    "REVOLUTION_CHANGE": {
        "label": "Revolution Change",
        "visible": ["filter"],
        "json_template": {"type": "REVOLUTION_CHANGE"}
    },
    "MEKRAID": {
        "label": "Mekraid",
        "visible": ["value1", "filter"],
        "label_value1": "Level",
        "json_template": {"type": "MEKRAID"}
    },
    "MOVE_UNDER": {
        "label": "Move Card Under",
        "visible": ["scope", "filter", "value1"],
        "json_template": {"type": "MOVE_TO_UNDER_CARD"}
    },
    "RESET_INSTANCE": {
        "label": "Reset (Devolve/Un-link)",
        "visible": ["scope", "filter"],
        "json_template": {"type": "RESET_INSTANCE"}
    },

    # --- LOGIC & SPECIAL ---
    "SUMMON_TOKEN": {
        "label": "Summon Token",
        "visible": ["value1", "str_val"],
        "label_value1": "Count",
        "label_str_val": "Token ID",
        "json_template": {"type": "SUMMON_TOKEN"}
    },
    "COUNT_CARDS": {
        "label": "Count Cards",
        "visible": ["scope", "filter", "output_value_key"],
        "produces_output": True,
        "json_template": {"type": "COUNT_CARDS"}
    },
    "GET_GAME_STAT": {
        "label": "Get Game Stat",
        "visible": ["str_val", "output_value_key"],
        "label_str_val": "Stat Key",
        "produces_output": True,
        "json_template": {"type": "GET_GAME_STAT"}
    },
    "SELECT_NUMBER": {
        "label": "Select Number",
        "visible": ["value1", "output_value_key"],
        "label_value1": "Max",
        "produces_output": True,
        "json_template": {"type": "SELECT_NUMBER"}
    },
    "SELECT_OPTION": {
        "label": "Select Option",
        "visible": ["value1", "value2", "str_val"],
        "json_template": {"type": "SELECT_OPTION"}
    },
    "FRIEND_BURST": {
        "label": "Friend Burst",
        "visible": ["filter"],
        "json_template": {"type": "FRIEND_BURST"}
    },
    "CAST_SPELL": {
        "label": "Cast Spell (Effect)",
        "visible": ["scope", "filter"],
        "json_template": {"type": "CAST_SPELL"}
    }
}
