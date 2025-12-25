import json
from typing import Any


# Minimal helper to export a snapshot of GameState from Python bindings or from a serialized file.
# For now, the project uses C++ engine and Python bindings; this script accepts a dm_ai_module.GameState-like dict and writes JSON.

def export_game_state(gs_obj: Any, out_path: str) -> None:
    # gs_obj is expected to be a dict-like structure with players and pending_effects
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(gs_obj, f, indent=2)

if __name__ == '__main__':
    print('This helper expects to be called from Python code that has a GameState dict.')
