import os
import importlib

# Force headless fallback for tests
os.environ['DM_EDITOR_HEADLESS'] = '1'

from dm_toolkit.gui.editor.forms.parts.cost_reduction_editor import CostReductionEditor


def test_preview_basic_units_and_limits():
    ed = CostReductionEditor()
    ed.set_value([
        {"type": "ACTIVE_PAYMENT", "amount": 3, "unit_cost": 2, "min_mana_cost": 1, "max_units": 5}
    ])
    # default uses entry amount (3)
    assert ed.compute_effective_cost() == 6
    assert "6" in ed.get_preview_text()

    # explicit units larger than max_units clamps to max_units
    assert ed.compute_effective_cost(units=10) == 10


def test_preview_min_mana_only():
    ed = CostReductionEditor()
    ed.set_value([
        {"type": "PASSIVE", "amount": 1, "min_mana_cost": 2}
    ])
    # unit_cost missing -> base_cost 0 -> effective becomes min_mana_cost
    assert ed.compute_effective_cost() == 2
    assert ed.get_preview_text() == "Preview (1 unit): 2"
