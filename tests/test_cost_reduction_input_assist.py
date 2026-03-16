import os

# force headless
os.environ['DM_EDITOR_HEADLESS'] = '1'

from dm_toolkit.gui.editor.forms.parts.cost_reduction_editor import CostReductionEditor


def test_input_assist_defaults_and_derivation():
    ed = CostReductionEditor()
    # empty -> defaults
    ed.set_value([])
    assert ed.suggest_input_assist() == {"unit_cost": 1, "max_units": 1, "min_mana_cost": 0}

    # amount-only entry
    ed.set_value([{"type": "ACTIVE_PAYMENT", "amount": 4}])
    hints = ed.suggest_input_assist()
    assert hints['unit_cost'] == 1
    assert hints['max_units'] == 4
    assert hints['min_mana_cost'] == (1 * 4) // 2

    # explicit unit_cost and max_units
    ed.set_value([{"type": "ACTIVE_PAYMENT", "amount": 2, "unit_cost": 3, "max_units": 5}])
    hints = ed.suggest_input_assist()
    assert hints['unit_cost'] == 3
    assert hints['max_units'] == 5
    assert hints['min_mana_cost'] == (3 * 5) // 2
