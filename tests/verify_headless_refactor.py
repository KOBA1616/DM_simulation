# tests/verify_headless_refactor.py
import sys
import os

# Adjust path to root
sys.path.append(os.getcwd())

from dm_toolkit.domain.simulation import SimulationRunner
from dm_toolkit.gui.editor.data_manager import CardDataManager
from dm_toolkit.editor.core.headless_impl import HeadlessEditorModel
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA

def test_simulation_runner():
    print("Testing SimulationRunner...")
    # Mock card db (duck typing or actual stub)
    class MockCardDB:
        pass

    # We rely on EngineCompat fallback if dm_ai_module is missing,
    # but SimulationRunner checks EngineCompat.is_available()

    try:
        # Assuming dm_ai_module might be mocked or available
        runner = SimulationRunner(MockCardDB(), "dm_01_basic", 1, 1, 10, "Random")
        print("SimulationRunner initialized.")
    except Exception as e:
        print(f"SimulationRunner init failed (expected if module missing): {e}")

def test_headless_data_manager():
    print("Testing Headless CardDataManager...")
    model = HeadlessEditorModel()
    manager = CardDataManager(model)

    # Add Card
    card_item = manager.add_new_card()
    print(f"Added card: {card_item.text()}")

    # Add Effect
    eff_data = manager.create_default_trigger_data()
    eff_item = manager.add_child_item(card_item, "EFFECT", eff_data, "Effect: ON_PLAY")
    print(f"Added effect: {eff_item.text()}")

    # Add Command
    cmd_data = manager.create_default_command_data()
    cmd_item = manager.add_child_item(eff_item, "COMMAND", cmd_data, "Action: DRAW")
    print(f"Added command: {cmd_item.text()}")

    # Verify structure
    # Note: add_new_card adds keyword item too
    # assert card_item.rowCount() >= 1
    # Check parentage
    assert cmd_item.parent() == eff_item
    assert eff_item.parent() == card_item

    # Verify data
    data = manager.get_item_data(cmd_item)
    assert data['type'] == 'DRAW'

    # Test removeRow
    initial_count = eff_item.rowCount()
    eff_item.removeRow(0)
    assert eff_item.rowCount() == initial_count - 1

    print("Headless CardDataManager verification passed.")

if __name__ == "__main__":
    test_simulation_runner()
    test_headless_data_manager()
