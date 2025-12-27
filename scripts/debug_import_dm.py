import importlib
m = importlib.import_module('dm_ai_module')
print('module repr:', repr(m))
print('module file:', getattr(m, '__file__', None))
print('has GameState:', hasattr(m, 'GameState'))
print('has _CARD_REGISTRY:', hasattr(m, '_CARD_REGISTRY'))
print('module type:', type(m))
print('keys sample:', list(m.__dict__.keys())[:40])
