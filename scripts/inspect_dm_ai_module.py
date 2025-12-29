import dm_ai_module
import pprint
print('dm_ai_module exports:')
print([n for n in dir(dm_ai_module) if not n.startswith('_')])
