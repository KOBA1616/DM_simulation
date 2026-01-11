# -*- coding: utf-8 -*-

class EffectDef:
    """
    Configuration class for Effect/Ability Types.
    Defines which fields are visible for a given effect type.
    """
    def __init__(self, key, mode="TRIGGERED", visible=None, labels=None):
        self.key = key
        self.mode = mode  # "TRIGGERED" or "STATIC"
        self.visible = visible or []
        self.labels = labels or {}

    def build_config(self):
        conf = {
            "mode": self.mode,
            "visible": self.visible,
        }
        for k, v in self.labels.items():
            if k == "value": conf["label_value"] = v
            elif k == "str_val": conf["label_str_val"] = v
        return conf

# Field constants for readability
# TRIGGERED fields
COND = "condition"

# STATIC fields
TYPE = "layer_type" # Usually implicit
VAL = "value"
STR = "str_val"
KW = "keyword"
FILT = "filter"

_definitions = [
    # --- Triggers ---
    # Most triggers just need a condition. Actions are added as children.
    EffectDef("ON_PLAY", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_ATTACK", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_BLOCK", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_DESTROY", mode="TRIGGERED", visible=[COND]),
    EffectDef("TURN_START", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_OPPONENT_DRAW", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_CAST_SPELL", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_ATTACK_FROM_HAND", mode="TRIGGERED", visible=[COND]),
    EffectDef("AT_BREAK_SHIELD", mode="TRIGGERED", visible=[COND]),
    EffectDef("ON_OTHER_ENTER", mode="TRIGGERED", visible=[COND]),

    # PASSIVE_CONST as a Trigger implies it holds actions (like Keywords) as children
    EffectDef("PASSIVE_CONST", mode="TRIGGERED", visible=[COND]),

    # --- Static Layers ---
    EffectDef("COST_MODIFIER", mode="STATIC",
              visible=[TYPE, VAL, FILT, COND],
              labels={"value": "Cost Reduction"}),

    EffectDef("POWER_MODIFIER", mode="STATIC",
              visible=[TYPE, VAL, FILT, COND],
              labels={"value": "Power Adjustment"}),

    EffectDef("GRANT_KEYWORD", mode="STATIC",
              visible=[TYPE, KW, FILT, COND]),

    EffectDef("SET_KEYWORD", mode="STATIC",
              visible=[TYPE, KW, FILT, COND]),
]

EFFECT_UI_CONFIG = {}
for eff in _definitions:
    EFFECT_UI_CONFIG[eff.key] = eff.build_config()

# Group lists for the UI
TRIGGER_TYPES = [d.key for d in _definitions if d.mode == "TRIGGERED"]
LAYER_TYPES = [d.key for d in _definitions if d.mode == "STATIC"]
