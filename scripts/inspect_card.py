import json, dm_ai_module, pathlib
card = {
    "id": 3000,
    "name": "MetaCard",
    "civilizations": [],
    "type": 0,
    "cost": 3,
    "power": 0,
    "races": [],
    "effects": [],
    "metamorph_abilities": [
        {
            "trigger": 0,
            "condition": None,
            "actions": [
                {"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False}
            ]
        }
    ]
}
path = pathlib.Path("C:/Windows/Temp/cards_auto_test.json")
path.write_text(json.dumps([card]), encoding='utf-8')
loader = dm_ai_module.JsonLoader
card_map = loader.load_cards(str(path))
entries = list(card_map.values()) if hasattr(card_map, 'values') else list(card_map)
first = entries[0]
print('TYPE:', type(first))
print('DIR:', dir(first))
try:
    print('HAS metamorph_abilities?', hasattr(first, 'metamorph_abilities'))
    try:
        print('metamorph_abilities repr:', getattr(first, 'metamorph_abilities'))
    except Exception as e:
        print('cannot access metamorph_abilities:', e)
    try:
        print('repr(first):', repr(first))
    except Exception as e:
        print('repr error:', e)
except Exception as e:
    print('error checking attrs', e)
