import os
os.environ['DM_DISABLE_NATIVE'] = '1'
import dm_ai_module
names = [k for k in dir(dm_ai_module) if not k.startswith('_')]
print('HAS_GameInstance', hasattr(dm_ai_module,'GameInstance'))
print('HAS_GameResult', hasattr(dm_ai_module,'GameResult'))
print('SAMPLE_NAMES', names[:200])
