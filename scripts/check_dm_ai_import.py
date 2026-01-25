import dm_ai_module
import sys

try:
    print('file:', getattr(dm_ai_module, '__file__', None))
    print('GameState:', hasattr(dm_ai_module, 'GameState'))
    print('JsonLoader:', hasattr(dm_ai_module, 'JsonLoader'))
    print('PhaseManager:', hasattr(dm_ai_module, 'PhaseManager'))
    print('CommandSystem:', hasattr(dm_ai_module, 'CommandSystem'))
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
