import os, json, sys, tempfile
os.environ['DM_DISABLE_NATIVE']='1'
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import dm_ai_module as dm
print('LOADED DM MODULE:', getattr(dm, '__file__', None))
cards = [
    {"id":9101,"name":"Target Creature","type":"CREATURE","cost":5,"power":500,"civilizations":["NATURE"]},
    {"id":9102,"name":"Support Static","type":"CREATURE","cost":1,"power":300,"civilizations":["NATURE"],"static_abilities":[{"type":"COST_MODIFIER","value_mode":"STAT_SCALED","stat_key":"CREATURES_PLAYED","per_value":1,"min_stat":1,"max_reduction":3}]},
    {"id":9103,"name":"Mana","type":"MANA","civilizations":["NATURE"]}
]
fd, path = tempfile.mkstemp(prefix='mini_stat_', suffix='.json', text=True)
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    json.dump(cards, f)
print('WROTE', path)
with open(path, 'r', encoding='utf-8') as f:
    cont = f.read()
    print('FILE CONTENT:', cont)
db = dm.JsonLoader.load_cards(path)
print('DB_KEYS:', list(db.keys()))
if 9102 in db:
    print('STATIC_ABS TYPE:', type(db[9102].static_abilities))
    try:
        print('STATIC_ABS LEN:', len(db[9102].static_abilities))
        print('STATIC_ABS:', db[9102].static_abilities)
    except Exception as e:
        print('STATIC_ABS: unreadable', e)
