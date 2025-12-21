import sys
import os

if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(r"C:\Program Files (x86)\mingw64\bin")
    except Exception:
        pass
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dm_ai_module
import os

print("Module loaded successfully")
gs = dm_ai_module.GameState(123)
print(f"Turn: {gs.turn_number}")

# Test Tensor Converter (provide empty card_db mapping)
tensor = dm_ai_module.TensorConverter.convert_to_tensor(gs, 0, {})
print(f"Tensor size: {len(tensor)}")
print(f"Expected size: {dm_ai_module.TensorConverter.INPUT_SIZE}")
