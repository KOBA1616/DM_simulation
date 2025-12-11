# Simple localization placeholder. Keep ASCII to avoid encoding issues.
TRANSLATIONS = {
    # Action Types
    "GRANT_KEYWORD": "Grant Keyword",
    "MOVE_CARD": "Move Card",
    "FRIEND_BURST": "Friend Burst",

    # Zones
    "HAND": "Hand",
    "BATTLE_ZONE": "Battle Zone",
    "GRAVEYARD": "Graveyard",
    "MANA_ZONE": "Mana Zone",
    "SHIELD_ZONE": "Shield Zone",
    "DECK_BOTTOM": "Deck Bottom",
    "DECK_TOP": "Deck Top",

    # Labels
    "Destination Zone": "Destination Zone",
    "Keyword": "Keyword",
    "Duration (Turns)": "Duration (Turns)",
    "Race (e.g. Fire Bird)": "Race (e.g. Fire Bird)"
}


def translate(key: str) -> str:
    """Return localized text when available, otherwise echo the key."""
    return TRANSLATIONS.get(key, key)


def tr(text: str) -> str:
    return TRANSLATIONS.get(text, text)
