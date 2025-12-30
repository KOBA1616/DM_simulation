
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
    if hasattr(dm_ai_module, 'ActionEncoder'):
        print(f"ActionEncoder found. TOTAL_ACTION_SIZE: {getattr(dm_ai_module.ActionEncoder, 'TOTAL_ACTION_SIZE', 'Not Found')}")
    else:
        print("ActionEncoder NOT found in dm_ai_module")
except ImportError:
    print("Could not import dm_ai_module")
