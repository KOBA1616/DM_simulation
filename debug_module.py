import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'dm_toolkit'))

try:
    import dm_ai_module
    print(f"Module file: {dm_ai_module.__file__}")
    print("Module attributes:")
    for attr in dir(dm_ai_module):
        print(f"  {attr}")
    
    if hasattr(dm_ai_module, 'GameInstance'):
        gi = dm_ai_module.GameInstance(0)
        print("GameInstance attributes:")
        for attr in dir(gi):
            if not attr.startswith('__'):
                print(f"  {attr}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
