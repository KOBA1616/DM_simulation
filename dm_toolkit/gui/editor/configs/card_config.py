# -*- coding: utf-8 -*-
# Card Data Configuration
from dm_toolkit.gui.localization import tr

CARD_TYPES = [
    "CREATURE",
    "SPELL",
    "EVOLUTION_CREATURE",
    "TAMASEED",
    "CASTLE",
    "NEO_CREATURE",
    "G_NEO_CREATURE"
]

CIVILIZATIONS = [
    "LIGHT",
    "WATER",
    "DARKNESS",
    "FIRE",
    "NATURE",
    "ZERO",
    "COLORLESS"
]

RACE_EXAMPLES = [
    "Dragon",
    "Fire Bird",
    "Cyber Lord",
    "Demon Command",
    "Angel Command",
    "Liquid People",
    "Beast Folk",
    "Human",
    "Machine Eater"
]

# Mapping of Card Type to UI visibility flags
# { Type: { 'power': bool, 'evolution_cond': bool } }
CARD_TYPE_UI_CONFIG = {
    "CREATURE": { "power": True, "evolution_cond": False },
    "SPELL": { "power": False, "evolution_cond": False },
    "EVOLUTION_CREATURE": { "power": True, "evolution_cond": True },
    "TAMASEED": { "power": False, "evolution_cond": False },
    "CASTLE": { "power": False, "evolution_cond": False },
    "NEO_CREATURE": { "power": True, "evolution_cond": True },
    "G_NEO_CREATURE": { "power": True, "evolution_cond": True },
}

def get_card_ui_config(card_type):
    return CARD_TYPE_UI_CONFIG.get(card_type, { "power": True, "evolution_cond": False })
