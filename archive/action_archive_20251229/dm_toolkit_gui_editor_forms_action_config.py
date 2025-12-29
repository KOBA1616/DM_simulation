# Archived copy of dm_toolkit/gui/editor/forms/action_config.py
# Original preserved on 2025-12-29

"""
Archive: dm_toolkit/gui/editor/forms/action_config.py
"""

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

_definitions = [
    # reduced for archive
]

ACTION_CATEGORIES: Dict[str, Dict[str, Any]] = {}
ACTION_UI_CONFIG: Dict[str, Dict[str, Any]] = {}
