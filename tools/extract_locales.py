
import sys
import os
import json

# Mock dm_ai_module if it doesn't exist to avoid import errors
# We just need the TRANSLATIONS dict which is populated with string keys first.
# The dynamic enum part requires the module, but we can't easily serialize Enums to JSON anyway without their string representation.
# So we will focus on extracting the string keys defined in the literal dict.

# Add python directory to path to find dm_toolkit
sys.path.append(os.path.abspath('python'))

# We will read the file manually to extract the TRANSLATIONS dict literal
# because importing it might run the code that depends on dm_ai_module which might not match the static strings we want to externalize.
# Actually, importing is fine if we just filter for string keys.

try:
    from dm_toolkit.gui import localization
except ImportError:
    # If import fails (e.g. deps), fallback to manual parsing?
    # Let's try to assume we can import it.
    # If dm_ai_module is missing, localization.py handles it gracefully.
    pass

def extract():
    from dm_toolkit.gui import localization

    data = {}
    for k, v in localization.TRANSLATIONS.items():
        if isinstance(k, str):
            data[k] = v

    # Also manual check for the ones that might be enums in the dict but we want to capture their string key intention if possible?
    # No, the TRANSLATIONS literal in the file uses string keys for most things.
    # The code `TRANSLATIONS[member] = ...` adds Enum keys.
    # The code `TRANSLATIONS[member.name] = TRANSLATIONS[member]` adds string keys for those enums.

    # We just want the strings.

    with open('data/locales/ja.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, sort_keys=True)

    print(f"Extracted {len(data)} keys to data/locales/ja.json")

if __name__ == "__main__":
    extract()
