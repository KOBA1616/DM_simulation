import sys, os
sys.path.append(os.path.join(os.getcwd(), 'bin'))
import dm_ai_module
print('CommandType members:')
print(list(dm_ai_module.CommandType.__members__.keys()))
print('Zone members:')
print(list(dm_ai_module.Zone.__members__.keys()) if hasattr(dm_ai_module,'Zone') else 'No Zone')
print('TargetScope members:')
print(list(dm_ai_module.TargetScope.__members__.keys()))
