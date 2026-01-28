
import sys
import os

try:
    import dm_ai_module
    print(f"dm_ai_module loaded. IS_NATIVE: {getattr(dm_ai_module, 'IS_NATIVE', False)}")
except ImportError:
    print("Could not import dm_ai_module")
    sys.exit(1)

print("\n--- MCTS Inspection ---")
if hasattr(dm_ai_module, 'MCTS'):
    print(f"MCTS found: {dm_ai_module.MCTS}")
    print("Methods:")
    print(dir(dm_ai_module.MCTS))
    try:
        print("\nInit Docstring:")
        print(dm_ai_module.MCTS.__init__.__doc__)
    except:
        pass
else:
    print("MCTS class NOT found in dm_ai_module")

print("\n--- TensorConverter Inspection ---")
if hasattr(dm_ai_module, 'TensorConverter'):
    print(f"TensorConverter found: {dm_ai_module.TensorConverter}")
    try:
        print("\nInit Docstring:")
        print(dm_ai_module.TensorConverter.__init__.__doc__)
    except:
        pass
else:
    print("TensorConverter class NOT found in dm_ai_module")

print("\n--- NeuralNetwork/Model Inspection ---")
for attr in dir(dm_ai_module):
    if 'Network' in attr or 'Model' in attr:
        print(f"Found candidate: {attr}")
