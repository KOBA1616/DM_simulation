import os
os.environ['DM_DISABLE_NATIVE'] = '1'
from dm_toolkit import dm_ai_module
from dm_toolkit.dm_ai_module import GameInstance, Phase

g = GameInstance()
print('before start:', g.state.current_phase)
g.start_game()
print('after start:', g.state.current_phase, type(g.state.current_phase))
print('active player:', g.state.active_player_id)
