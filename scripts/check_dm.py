import dm_ai_module
print('IS_NATIVE=', getattr(dm_ai_module,'IS_NATIVE', None))
print('file=', getattr(dm_ai_module,'__file__', None))
print('Available symbols:', [k for k in dir(dm_ai_module) if not k.startswith('_')][:40])
