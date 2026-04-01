from enum import Enum

class SemanticMetadataFlags(Enum):
    TARGETS = "targets"
    DRAWS = "draws"
    DISCARDS = "discards"
    DESTROYS = "destroys"
    SHIELDS_ADDED = "shields_added"
    SHIELDS_BROKEN = "shields_broken"
    MANA_CHARGED = "mana_charged"
    COSTS_REDUCED = "costs_reduced"
    GRANTS_SPEED_ATTACKER = "grants_speed_attacker"
    GRANTS_BLOCKER = "grants_blocker"
    RETURNS_TO_HAND = "returns_to_hand"
    ON_CAST_TRIGGER = "on_cast_trigger"
