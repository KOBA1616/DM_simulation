import dm_ai_module
import inspect
print('module file:', getattr(dm_ai_module, '__file__', None))
print('has set_batch_callback:', hasattr(dm_ai_module, 'set_batch_callback'))
print('set_batch_callback repr:', repr(getattr(dm_ai_module, 'set_batch_callback', None)))
print('has has_batch_callback:', hasattr(dm_ai_module, 'has_batch_callback'))
print('has clear_batch_callback:', hasattr(dm_ai_module, 'clear_batch_callback'))
print('has NeuralEvaluator:', hasattr(dm_ai_module, 'NeuralEvaluator'))
print('has ActionEncoder:', hasattr(dm_ai_module, 'ActionEncoder'))
print('dir sample (subset):', [n for n in dir(dm_ai_module) if 'batch' in n or 'Neural' in n or 'Action' in n or 'evaluate' in n or 'GameState' in n])
