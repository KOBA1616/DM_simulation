import json
from dm_ai_module import JsonLoader
sample = [{"id":9999,"effects":[{"trigger":"ON_PLAY","actions":[{"type":"DRAW_CARD","value1":2}]}]}]
open('tmp_dummy.json','w', encoding='utf-8').write(json.dumps(sample))
cards=JsonLoader.load_cards('tmp_dummy.json')
print('cards keys', list(cards.keys()))
cd=cards.get(9999)
print('effects len', len(cd.effects))
print('first effect actions', cd.effects[0].actions)
print('first effect commands', cd.effects[0].commands)
