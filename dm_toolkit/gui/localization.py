# Simple localization placeholder. Keep ASCII to avoid encoding issues.
TRANSLATIONS = {}


def translate(key: str) -> str:
    """Return localized text when available, otherwise echo the key."""
    return TRANSLATIONS.get(key, key)


def tr(text: str) -> str:
    return TRANSLATIONS.get(text, text)
