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
# Must unwrap the GameStateWrapper before passing to C++ binding
native_gs = getattr(gs, '_native', gs)
# The binding expects a CardDatabase object, not a dict
card_db = dm_ai_module.CardDatabase()
tensor = dm_ai_module.TensorConverter.convert_to_tensor(native_gs, 0, card_db)
print(f"Tensor size: {len(tensor)}")
print(f"Expected size: {dm_ai_module.TensorConverter.INPUT_SIZE}")
