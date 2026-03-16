import os


def test_input_assist_updates_fields():
    # Use headless fallback
    os.environ['DM_EDITOR_HEADLESS'] = '1'
    from dm_toolkit.gui.editor.forms.parts.cost_reduction_editor import CostReductionEditor

    w = CostReductionEditor()
    sample = [{"type": "ACTIVE_PAYMENT", "amount": 1}]
    w.set_value(sample)
    # ensure initial
    out = w.get_value()
    assert out[0].get('amount') == 1

    # update selected fields
    w.set_selected_index(0)
    w.update_selected_fields(amount=5, min_mana_cost=2, unit_cost=3, max_units=4)

    out2 = w.get_value()
    assert out2[0]['amount'] == 5
    assert out2[0]['min_mana_cost'] == 2
    assert out2[0]['unit_cost'] == 3
    assert out2[0]['max_units'] == 4
