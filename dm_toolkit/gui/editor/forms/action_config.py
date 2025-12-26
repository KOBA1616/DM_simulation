# -*- coding: utf-8 -*-
from typing import Dict, List, Any, Optional

# Define Category Metadata (Order and Labels)
CATEGORY_METADATA = {
    "ZONE_CONTROL": "Zone Control (Move)",
    "STATE_MUTATION": "State & Mutation",
    "SEARCH_REVEAL": "Search & Reveal",
    "MULTILAYER": "Multilayer Card Operations",
    "LOGIC_SPECIAL": "Logic & Special"
}

class ActionDef:
    """
    Configuration class for an Action Type.
    Separates the definition logic from the generated configuration dictionary.
    """
    def __init__(self, key: str, label: str, category: str, visible: Optional[List[str]] = None, labels: Optional[Dict[str, str]] = None,
                 produces_output: bool = False, can_be_optional: bool = False,
                 json_template: Optional[Dict[str, Any]] = None, output_label: Optional[str] = None):
        self.key = key
        self.label = label
        self.category = category
        self.visible = visible or []
        self.labels = labels or {}
        self.produces_output = produces_output
        self.can_be_optional = can_be_optional
        self.json_template = json_template or {}
        self.output_label = output_label

    def build_config(self):
        """Generates the dictionary entry for ACTION_UI_CONFIG."""
        conf = {
            "label": self.label,
            "visible": self.visible,
            "produces_output": self.produces_output,
            "can_be_optional": self.can_be_optional,
            "json_template": self.json_template
        }
        if self.output_label:
            conf["outputs"] = {"output_value_key": self.output_label}

        for k, v in self.labels.items():
            # Map friendly label names to config keys
            if k == "value1": conf["label_value1"] = v
            elif k == "value2": conf["label_value2"] = v
            elif k == "str_val": conf["label_str_val"] = v

        return conf

# List of Action Definitions
_definitions = [
    # --- ZONE CONTROL ---
    ActionDef("DRAW_CARD", "Draw Card", "ZONE_CONTROL",
              visible=["value1"], labels={"value1": "Count"},
              produces_output=True, output_label="Cards Drawn",
              json_template={"type": "MOVE_CARD", "source_zone": "DECK", "destination_zone": "HAND"}),

    ActionDef("DESTROY", "Destroy", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice"],
              can_be_optional=True,
              json_template={"type": "MOVE_CARD", "destination_zone": "GRAVEYARD", "source_zone": "BATTLE_ZONE"}),

    ActionDef("SEND_TO_MANA", "Send to Mana", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "source_zone"],
              can_be_optional=True,
              json_template={"type": "MOVE_CARD", "destination_zone": "MANA_ZONE"}),

    ActionDef("BOUNCE", "Return to Hand (Bounce)", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "source_zone"],
              can_be_optional=True,
              json_template={"type": "MOVE_CARD", "destination_zone": "HAND"}),

    ActionDef("SHIELD_BREAK", "Shield Break (Triggers OK)", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "value1"], labels={"value1": "Count (if random/top)"},
              json_template={"type": "MOVE_CARD", "source_zone": "SHIELD_ZONE", "destination_zone": "HAND", "str_val": "BREAK"}),

    ActionDef("ADD_SHIELD_TO_HAND", "Add Shield to Hand (No Triggers)", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "value1"], labels={"value1": "Count"},
              json_template={"type": "MOVE_CARD", "source_zone": "SHIELD_ZONE", "destination_zone": "HAND", "str_val": "RETURN"}),

    ActionDef("ADD_SHIELD", "Add Shield", "ZONE_CONTROL",
              visible=["scope", "filter", "value1", "source_zone"], labels={"value1": "Count"},
              json_template={"type": "ADD_SHIELD"}),

    ActionDef("DISCARD", "Discard", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "value1"], labels={"value1": "Count (Random/Select)"},
              produces_output=True,
              json_template={"type": "MOVE_CARD", "source_zone": "HAND", "destination_zone": "GRAVEYARD"}),

    ActionDef("PUT_INTO_BATTLE", "Put into Battle", "ZONE_CONTROL",
              visible=["source_zone", "filter", "scope"],
              json_template={"type": "MOVE_CARD", "destination_zone": "BATTLE_ZONE"}),

    ActionDef("SEND_TO_DECK", "Send to Deck", "ZONE_CONTROL",
              visible=["scope", "filter", "target_choice", "str_val"], labels={"str_val": "Position (TOP/BOTTOM)"},
              json_template={"type": "MOVE_CARD", "destination_zone": "DECK"}),

    # --- STATE & MUTATION ---
    ActionDef("MODIFY_POWER", "Modify Power", "STATE_MUTATION",
              visible=["scope", "filter", "value1", "value2"], labels={"value1": "Power Mod", "value2": "Duration (Turns)"},
              json_template={"type": "MODIFY_POWER"}),

    ActionDef("GRANT_ABILITY", "Grant Ability", "STATE_MUTATION",
              visible=["scope", "filter", "str_val", "value2"], labels={"str_val": "Keyword (e.g. blocker)", "value2": "Duration (Turns)"},
              json_template={"type": "GRANT_KEYWORD"}),

    ActionDef("TAP", "Tap", "STATE_MUTATION",
              visible=["scope", "filter", "target_choice"],
              json_template={"type": "MUTATE", "str_val": "TAP"}),

    ActionDef("UNTAP", "Untap", "STATE_MUTATION",
              visible=["scope", "filter", "target_choice"],
              json_template={"type": "MUTATE", "str_val": "UNTAP"}),

    ActionDef("SHIELD_BURN", "Shield Burn", "STATE_MUTATION",
              visible=["scope", "filter", "value1"], labels={"value1": "Count"},
              json_template={"type": "MUTATE", "str_val": "SHIELD_BURN"}),

    ActionDef("COST_REDUCTION", "Cost Reduction", "STATE_MUTATION",
              visible=["filter", "value1", "source_zone"], labels={"value1": "Reduction Amount"},
              json_template={"type": "APPLY_MODIFIER", "str_val": "COST"}),

    # --- SEARCH & REVEAL ---
    ActionDef("SEARCH_DECK", "Search Deck", "SEARCH_REVEAL",
              visible=["filter", "output_value_key", "value1"],
              produces_output=True,
              json_template={"type": "SEARCH_DECK"}),

    ActionDef("LOOK_AND_ADD", "Look & Add", "SEARCH_REVEAL",
              visible=["value1", "value2", "filter", "output_value_key"], labels={"value1": "Look N", "value2": "Add M"},
              produces_output=True,
              json_template={"type": "LOOK_AND_ADD"}),

    ActionDef("REVEAL_CARDS", "Reveal Cards", "SEARCH_REVEAL",
              visible=["scope", "filter", "value1"],
              json_template={"type": "REVEAL_CARDS"}),

    ActionDef("SHUFFLE_DECK", "Shuffle Deck", "SEARCH_REVEAL",
              visible=["scope"],
              json_template={"type": "SHUFFLE_DECK"}),

    # --- MULTILAYER ---
    ActionDef("REVOLUTION_CHANGE", "Revolution Change", "MULTILAYER",
              visible=["filter"],
              json_template={"type": "REVOLUTION_CHANGE"}),

    ActionDef("MEKRAID", "Mekraid", "MULTILAYER",
              visible=["value1", "filter"], labels={"value1": "Level"},
              json_template={"type": "MEKRAID"}),

    ActionDef("MOVE_UNDER", "Move Card Under", "MULTILAYER",
              visible=["scope", "filter", "value1"],
              json_template={"type": "MOVE_TO_UNDER_CARD"}),

    ActionDef("RESET_INSTANCE", "Reset (Devolve/Un-link)", "MULTILAYER",
              visible=["scope", "filter"],
              json_template={"type": "RESET_INSTANCE"}),

    # --- LOGIC & SPECIAL ---
    ActionDef("SUMMON_TOKEN", "Summon Token", "LOGIC_SPECIAL",
              visible=["value1", "str_val"], labels={"value1": "Count", "str_val": "Token ID"},
              json_template={"type": "SUMMON_TOKEN"}),

    ActionDef("COUNT_CARDS", "Count Cards", "LOGIC_SPECIAL",
              visible=["scope", "filter", "output_value_key"],
              produces_output=True,
              json_template={"type": "COUNT_CARDS"}),

    ActionDef("GET_GAME_STAT", "Get Game Stat", "LOGIC_SPECIAL",
              visible=["str_val", "output_value_key"], labels={"str_val": "Stat Key"},
              produces_output=True,
              json_template={"type": "GET_GAME_STAT"}),

    ActionDef("SELECT_NUMBER", "Select Number", "LOGIC_SPECIAL",
              visible=["value1", "output_value_key"], labels={"value1": "Max"},
              produces_output=True,
              json_template={"type": "SELECT_NUMBER"}),

    ActionDef("SELECT_OPTION", "Select Option", "LOGIC_SPECIAL",
              visible=["value1", "value2", "str_val"],
              json_template={"type": "SELECT_OPTION"}),

    ActionDef("FRIEND_BURST", "Friend Burst", "LOGIC_SPECIAL",
              visible=["filter"],
              json_template={"type": "FRIEND_BURST"}),

    ActionDef("CAST_SPELL", "Cast Spell (Effect)", "LOGIC_SPECIAL",
              visible=["scope", "filter"],
              json_template={"type": "CAST_SPELL"})
]

# Generate Dictionary Exports
ACTION_CATEGORIES: Dict[str, Dict[str, Any]] = {}
ACTION_UI_CONFIG: Dict[str, Dict[str, Any]] = {}

# Initialize Categories
for cat_key, cat_label in CATEGORY_METADATA.items():
    ACTION_CATEGORIES[cat_key] = {
        "label": cat_label,
        "actions": []
    }

for action in _definitions:
    # Populate UI Config
    ACTION_UI_CONFIG[action.key] = action.build_config()

    # Populate Category List
    if action.category in ACTION_CATEGORIES:
        ACTION_CATEGORIES[action.category]["actions"].append(action.key)

# Backfill a few legacy/engine action keys that may not have explicit
# ActionDef entries but are referenced by the engine enums.
_legacy_keys = [
    'TRANSITION', 'ADD_MANA', 'RETURN_TO_HAND', 'BREAK_SHIELD',
    'SEARCH_DECK_BOTTOM', 'PLAY_FROM_ZONE', 'SEND_SHIELD_TO_GRAVE', 'SEND_TO_DECK_BOTTOM'
]
for _k in _legacy_keys:
    if _k not in ACTION_UI_CONFIG:
        ACTION_UI_CONFIG[_k] = {
            "label": _k,
            "visible": [],
            "produces_output": False,
            "can_be_optional": False,
            "json_template": {}
        }
