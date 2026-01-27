import sys
sys.path.insert(0, '.')
import dm_ai_module as dm
print('OK dm loaded')
print('ActionEncoder TOTAL_ACTION_SIZE =', getattr(dm, 'ActionEncoder').TOTAL_ACTION_SIZE)
print('CommandEncoder TOTAL_COMMAND_SIZE =', dm.CommandEncoder.TOTAL_COMMAND_SIZE)
print('index_to_command(0)=', dm.CommandEncoder.index_to_command(0))
print('index_to_command(20)=', dm.CommandEncoder.index_to_command(20))
