# -*- coding: utf-8 -*-

class CommandDef:
    """
    Configuration class for a Command Type.
    Separates the definition logic from the generated configuration dictionary.
    """
    def __init__(self, key, visible=None, labels=None,
                 produces_output=False, output_label=None):
        self.key = key
        self.visible = visible or []
        self.labels = labels or {}
        self.produces_output = produces_output
        self.output_label = output_label

    def build_config(self):
        """Generates the dictionary entry for COMMAND_UI_CONFIG."""
        conf = {
            "visible": self.visible,
            "produces_output": self.produces_output
        }
        if self.output_label:
            conf["outputs"] = {"output_value_key": self.output_label}

        for k, v in self.labels.items():
            # Map friendly label names to config keys (Command specific)
            if k == "amount": conf["label_amount"] = v
            elif k == "mutation_kind": conf["label_mutation_kind"] = v
            elif k == "str_param": conf["label_str_param"] = v

        return conf

_definitions = [
    CommandDef("NONE", visible=[]),

    # --- Atomic Actions (Shortcuts) ---
    CommandDef("DRAW_CARD",
               visible=["target_group", "amount", "optional", "input_link"],
               labels={"amount": "Cards to Draw"},
               produces_output=True, output_label="Cards Drawn"),

    CommandDef("DISCARD",
               visible=["target_group", "target_filter", "amount", "optional", "input_link", "output_link"],
               labels={"amount": "Count"},
               produces_output=True, output_label="Discarded Cards"),

    CommandDef("DESTROY",
               visible=["target_group", "target_filter", "amount", "input_link"],
               labels={"amount": "Count (if selecting)"}),

    CommandDef("MANA_CHARGE",
               visible=["target_group", "target_filter", "amount", "input_link"],
               labels={"amount": "Count (if selecting)"}),

    CommandDef("TAP",
               visible=["target_group", "target_filter", "input_link"]),

    CommandDef("UNTAP",
               visible=["target_group", "target_filter", "input_link"]),

    CommandDef("RETURN_TO_HAND",
               visible=["target_group", "target_filter", "amount", "input_link"],
               labels={"amount": "Count (if selecting)"}),

    CommandDef("BREAK_SHIELD",
               visible=["target_group", "target_filter", "amount", "input_link"],
               labels={"amount": "Count"}),

    # --- Generalized Commands ---
    CommandDef("TRANSITION",
               visible=["target_group", "target_filter", "from_zone", "to_zone", "amount", "optional", "input_link", "output_link"],
               labels={"amount": "Count"},
               produces_output=True, output_label="Moved Cards"),

    CommandDef("MUTATE",
               visible=["target_group", "target_filter", "mutation_kind", "amount", "str_param"],
               labels={"mutation_kind": "Mutation Type", "amount": "Value / Duration", "str_param": "Extra Param"}),

    CommandDef("POWER_MOD",
               visible=["target_group", "target_filter", "amount"],
               labels={"amount": "Power Adjustment"}),

    CommandDef("ADD_KEYWORD",
               visible=["target_group", "target_filter", "mutation_kind", "amount"],
               labels={"mutation_kind": "Keyword", "amount": "Duration (Turns)"}),

    # --- Information / Flow ---
    CommandDef("QUERY",
               visible=["target_group", "target_filter", "query_mode", "output_link"],
               produces_output=True, output_label="Query Result"),

    CommandDef("FLOW",
               visible=["str_param"],
               labels={"str_param": "Flow Instruction"}),

    CommandDef("SEARCH_DECK",
               visible=["target_filter", "amount", "output_link"],
               labels={"amount": "Count"},
               produces_output=True, output_label="Found Cards"),

    CommandDef("SHIELD_TRIGGER",
               visible=["target_group"])
]

# Generate Dictionary Export
COMMAND_UI_CONFIG = {}

for cmd in _definitions:
    COMMAND_UI_CONFIG[cmd.key] = cmd.build_config()
