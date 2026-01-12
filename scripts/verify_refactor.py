import sys
import os

# Add root to sys.path
sys.path.append(os.getcwd())

try:
    from dm_toolkit.gui import localization
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Check symbols
if not hasattr(localization, 'TRANSLATIONS'):
    print("TRANSLATIONS missing")
    sys.exit(1)

if not hasattr(localization, 'tr'):
    print("tr missing")
    sys.exit(1)

if not hasattr(localization, 'get_card_civilizations'):
    print("get_card_civilizations missing")
    sys.exit(1)

if not hasattr(localization, 'describe_command'):
    print("describe_command missing")
    sys.exit(1)

# Check translation loading (check a known key)
key = "Destination Zone"
val = localization.tr(key)
if val == "移動先ゾーン":
    print(f"Translation check passed: '{key}' -> '{val}'")
else:
    print(f"Translation check failed: '{key}' -> '{val}' (Expected '移動先ゾーン')")
    sys.exit(1)

print("All checks passed")
