import os

import dm_ai_module


def test_tensor_converter_smoke():
    # Ensure MinGW DLL path is available on Windows when using MinGW-built extensions.
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(r"C:\Program Files (x86)\mingw64\bin")
        except Exception:
            pass

    gs = dm_ai_module.GameState(123)
    card_db = dm_ai_module.CardDatabase()

    tensor = dm_ai_module.TensorConverter.convert_to_tensor(gs, 0, card_db)
    assert isinstance(tensor, list)
    assert len(tensor) == dm_ai_module.TensorConverter.INPUT_SIZE
