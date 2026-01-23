import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import dm_ai_module as dm

print('Using python', sys.executable)
p = Path('data/cards.json')
if not p.exists():
    print('cards.json missing')
    sys.exit(1)

s = p.read_text(encoding='utf-8')
try:
    if hasattr(dm, 'CardRegistry') and hasattr(dm.CardRegistry, 'load_from_json'):
        dm.CardRegistry.load_from_json(s)
        print('CardRegistry.load_from_json succeeded')
    elif hasattr(dm, 'register_card_data'):
        # Fallback to register_card_data if provided
        try:
            dm.register_card_data(s)
            print('register_card_data succeeded')
        except Exception as e:
            print('register_card_data failed:', e)
    else:
        print('No CardRegistry.load_from_json or register_card_data available')
except Exception as e:
    print('Card registry load failed:', e)

gi = None
try:
    gi = dm.GameInstance(0)
    print('Constructed GameInstance(0)')
except Exception as e:
    print('GameInstance(0) construction failed:', e)

if gi is not None:
    try:
        gi.start_game()
        print('Called gi.start_game()')
        print('P0 shields:', len(gi.state.players[0].shield_zone))
        print('P0 hand:', len(gi.state.players[0].hand))
    except Exception as e:
        print('gi.start_game failed:', e)

try:
    if hasattr(dm, 'CardRegistry') and hasattr(dm.CardRegistry, 'get_all_definitions'):
        defs = dm.CardRegistry.get_all_definitions()
        print('CardRegistry.get_all_definitions count:', len(defs) if hasattr(defs, '__len__') else 'unknown')
    elif hasattr(dm, 'JsonLoader'):
        db = dm.JsonLoader.load_cards('data/cards.json')
        print('JsonLoader returned db size:', len(db) if hasattr(db, '__len__') else 'unknown')
except Exception as e:
    print('Failed to inspect registered card DB:', e)
