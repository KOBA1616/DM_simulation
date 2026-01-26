import sys
import os
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from dm_toolkit.editor.core.headless_impl import HeadlessEditorModel
from dm_toolkit.gui.editor.data_manager import CardDataManager
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.domain.simulation import SimulationRunner

def verify_headless_editor():
    print("Verifying Headless Editor...")
    model = HeadlessEditorModel()
    data_manager = CardDataManager(model)

    # 1. Add Card
    card_item = data_manager.add_new_card()
    if not card_item:
        print("FAIL: add_new_card returned None")
        return False

    print(f"Card Added: {card_item.text()}")

    # 2. Add Effect
    # Create default trigger data
    trig_data = data_manager.create_default_trigger_data()
    eff_item = data_manager.add_child_item(card_item, "EFFECT", trig_data, "Effect: ON_PLAY")
    if not eff_item:
        print("FAIL: add_child_item (Effect) returned None")
        return False

    print(f"Effect Added: {eff_item.text()}")

    # 3. Add Command
    cmd_data = data_manager.create_default_command_data()
    cmd_item = data_manager.add_child_item(eff_item, "COMMAND", cmd_data, "Action: DRAW")
    if not cmd_item:
        print("FAIL: add_child_item (Command) returned None")
        return False

    print(f"Command Added: {cmd_item.text()}")

    # 4. Verify Structure
    # Card is appended to root
    if model.root_item().row_count() != 1:
        print(f"FAIL: Root row count mismatch. Expected 1, got {model.root_item().row_count()}")
        return False

    c = model.root_item().child(0)
    # Card should have Keywords child + Effect child.
    if c.row_count() < 2:
        print(f"FAIL: Card child count mismatch. Expected >= 2, got {c.row_count()}")
        return False

    print("Headless Editor Verification PASSED")
    return True

def verify_simulation_runner():
    print("Verifying Simulation Runner...")
    # Just instantiate it
    try:
        runner = SimulationRunner(
            card_db=None,
            scenario_name="test_scenario",
            episodes=10,
            threads=1,
            sims=10,
            evaluator_type="Random"
        )
        print("SimulationRunner instantiated successfully.")
    except Exception as e:
        print(f"FAIL: SimulationRunner instantiation failed: {e}")
        return False

    print("Simulation Runner Verification PASSED")
    return True

if __name__ == "__main__":
    if verify_headless_editor() and verify_simulation_runner():
        print("ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("CHECKS FAILED")
        sys.exit(1)
