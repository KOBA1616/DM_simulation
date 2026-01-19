import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import dm_ai_module as dm


a = dm.Action()
print('Action instance dir:')
print([x for x in dir(a) if not x.startswith('_')])
print('\nAttribute values:')
for name in ('type','target_player','source_instance_id','card_id','slot_index','value1','target_shield','target_id','shield_instance','slot'):
    print(name, getattr(a, name, '<missing>'))

print('\nModule enums and sample types:')
for n in dir(dm):
    if any(k in n for k in ('ATTACK','BREAK','ActionType','PlayerIntent','BREAK_SHIELD')):
        print(n)

# Try to set some attributes and reprint
try:
    a.slot_index = 123
    a.value1 = 456
    a.source_instance_id = 789
    print('\nAfter setting slot_index/value1/source_instance_id:')
    print('slot_index', getattr(a,'slot_index','<missing>'))
    print('value1', getattr(a,'value1','<missing>'))
    print('source_instance_id', getattr(a,'source_instance_id','<missing>'))
except Exception as e:
    print('Setting attributes failed:', e)
