import os, json
os.environ['DM_DISABLE_NATIVE']='1'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import dm_ai_module as dm
cards = [
    {"id":9101,"name":"Target Creature","type":"CREATURE","cost":5,"power":500,"civilizations":["NATURE"]},
    {"id":9102,"name":"Support Static","type":"CREATURE","cost":1,"power":300,"civilizations":["NATURE"],"static_abilities":[{"type":"COST_MODIFIER","value_mode":"STAT_SCALED","stat_key":"CREATURES_PLAYED","per_value":1,"min_stat":1,"max_reduction":3}]},
    {"id":9103,"name":"Mana","type":"MANA","civilizations":["NATURE"]}
]
js = json.dumps(cards)
try:
    db = dm.JsonLoader.load_cards(js)
    print('DB_KEYS:', list(db.keys()))
except Exception as e:
    import traceback
    print('LOAD FAILED:', e)
    traceback.print_exc()
if 9102 in db:
    print('TYPE:', type(db[9102]))
    try:
        print('STATIC_ABS LEN:', len(db[9102].static_abilities))
        print('STATIC_ABS:', db[9102].static_abilities)
    except Exception as e:
        print('STATIC_ABS: unreadable', e)

# Inspect raw parse via json.loads
raw = json.loads(js)
print('RAW static:', raw[1].get('static_abilities'))
