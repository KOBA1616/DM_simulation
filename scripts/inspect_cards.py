import dm_ai_module
try:
    db=dm_ai_module.JsonLoader.load_cards('data/cards.json')
    print('len', len(db))
    for k,v in db.items():
        print('key',k,type(v))
        print([a for a in dir(v) if not a.startswith('_')][:40])
        break
except Exception as e:
    print('ERR',e)
