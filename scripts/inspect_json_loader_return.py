import json, tempfile
import dm_ai_module
from pathlib import Path

def make_legacy_card_json(card_id=1000):
    card = {"id": card_id, "name": "TestCard", "civilizations": [], "type": 0, "cost": 1, "power": 0, "races": [],
            "effects": [{"trigger": 0, "condition": None, "actions": [{"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False}]}]}
    return {"cards": [card]}

p = Path(tempfile.mkdtemp()) / 'cards.json'
with p.open('w', encoding='utf-8') as f:
    json.dump(make_legacy_card_json(), f)
loader = dm_ai_module.JsonLoader
cm = loader.load_cards(str(p))
print('TYPE:', type(cm))
print('REPR:', repr(cm))
try:
    print('DIR:', [x for x in dir(cm) if not x.startswith('_')])
except Exception as e:
    print('DIRERR', e)
# try common accessors
if hasattr(cm, 'values'):
    try:
        print('has values, len', len(list(cm.values())))
    except Exception as e:
        print('values err', e)
if hasattr(cm, 'get_all_cards'):
    try:
        gac = cm.get_all_cards()
        print('get_all_cards ->', type(gac), 'len', len(list(gac.values())))
    except Exception as e:
        print('get_all_cards err', e)
if hasattr(cm, 'cards'):
    try:
        print('cards attr type', type(cm.cards))
    except Exception as e:
        print('cards attr err', e)
print('DONE')
