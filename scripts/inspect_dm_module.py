import dm_ai_module as dm
import inspect

print('dm_ai_module file:', getattr(dm, '__file__', '<unknown>'))
print('IS_NATIVE:', getattr(dm, 'IS_NATIVE', False))
print('module attrs:', sorted([x for x in dir(dm) if not x.startswith('_')]))
cls = dm.CardDefinition
print('CardDefinition attrs:', sorted([x for x in dir(cls) if not x.startswith('_')]))
try:
	inst = cls(0, 'x', 'FIRE', [], 0, 0, dm.CardKeywords(), [])
	print('instance attrs:', sorted([x for x in dir(inst) if not x.startswith('_')]))
	print('has metamorph_abilities:', hasattr(inst, 'metamorph_abilities'))
except Exception as e:
	print('instantiation error:', e)
