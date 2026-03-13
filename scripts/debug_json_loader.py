import json, traceback
import dm_ai_module

def run():
    card = [{
        'id': 1000,
        'name': 'TestCard',
        'civilizations': [],
        'type': 0,
        'cost': 1,
        'power': 0,
        'races': [],
        'effects': [
            {
                'trigger': 0,
                'condition': None,
                'actions': [
                    {'type': 0, 'scope': 'SINGLE', 'filter': '', 'value1': 1, 'optional': False}
                ]
            }
        ]
    }]
    js = json.dumps(card)
    print('Calling load_cards_from_string...')
    try:
        res = dm_ai_module.JsonLoader.load_cards_from_string(js)
        print('Result length:', len(res))
    except Exception as e:
        print('Exception:')
        traceback.print_exc()

if __name__ == '__main__':
    run()
