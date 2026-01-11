# -*- coding: utf-8 -*-
"""
Metadata Registry for the Card Editor.
Centralizes lists of Types, Civilizations, and Command Groups.
"""
from dm_toolkit.consts import GRANTABLE_KEYWORDS as _KEYWORDS

# Re-export imported constants
GRANTABLE_KEYWORDS = _KEYWORDS

# Card Types
CARD_TYPES = [
    "CREATURE",
    "SPELL",
    "EVOLUTION_CREATURE",
    "TAMASEED",
    "CASTLE",
    "NEO_CREATURE",
    "G_NEO_CREATURE"
]

# Command Groups for UI organization
COMMAND_GROUPS = {
    'DRAW': [
        'DRAW_CARD'
    ],
    'CARD_MOVE': [
        'TRANSITION', 'RETURN_TO_HAND', 'DISCARD', 'DESTROY', 'MANA_CHARGE'
    ],
    'DECK_OPS': [
        'SEARCH_DECK', 'LOOK_AND_ADD', 'REVEAL_CARDS', 'SHUFFLE_DECK'
    ],
    'PLAY': [
        'PLAY_FROM_ZONE', 'CAST_SPELL'
    ],
    'BUFFER': [
        'LOOK_TO_BUFFER', 'SELECT_FROM_BUFFER', 'PLAY_FROM_BUFFER', 'MOVE_BUFFER_TO_ZONE'
    ],
    'CHEAT_PUT': [
        'MEKRAID', 'FRIEND_BURST', 'REVOLUTION_CHANGE'
    ],
    'GRANT': [
        'MUTATE', 'POWER_MOD', 'ADD_KEYWORD', 'TAP', 'UNTAP', 'REGISTER_DELAYED_EFFECT'
    ],
    'LOGIC': [
        'QUERY', 'FLOW', 'SELECT_NUMBER', 'CHOICE', 'SELECT_OPTION', 'IF', 'IF_ELSE', 'ELSE'
    ],
    'BATTLE': [
        'BREAK_SHIELD', 'RESOLVE_BATTLE', 'SHIELD_BURN', 'SHIELD_TRIGGER'
    ],
    'RESTRICTION': [
    ]
}
