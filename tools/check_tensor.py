import dm_ai_module as dm
from dm_ai_module import TensorConverter, GameState
s = GameState(0)
card_db = {}
vec = TensorConverter.convert_to_tensor(s, s.active_player_id, card_db)
print('len=', len(vec))
print('INPUT_SIZE expected=', dm.TensorConverter.INPUT_SIZE)
print('first 10:', vec[:10])
