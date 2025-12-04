
import sys
import os

# Add bin to path to simulate the environment
sys.path.append(os.path.join(os.getcwd(), 'bin'))
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import dm_ai_module
    print("SUCCESS: dm_ai_module imported successfully.")
    print(f"Docstring: {dm_ai_module.__doc__}")
except ImportError as e:
    print(f"FAILURE: ImportError: {e}")
except Exception as e:
    print(f"FAILURE: Unexpected error: {e}")
