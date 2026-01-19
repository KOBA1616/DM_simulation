import sys
sys.path.insert(0, '.')
import dm_ai_module as dm

print('CardInstance constructor test')
try:
    ci = dm.CardInstance()
    print('Created CardInstance empty:', type(ci))
    # Try setting attributes
    try:
        ci.card_id = 1
        ci.instance_id = 123
        print('Set card_id and instance_id')
    except Exception as e:
        print('Setting attrs failed:', e)
    gi = dm.GameInstance(0)
    p = gi.state.players[0]
    try:
        p.hand.append(ci)
        print('Appended to hand, hand size:', len(p.hand))
    except Exception as e:
        print('Append failed:', e)
except Exception as e:
    print('CardInstance creation failed:', e)
