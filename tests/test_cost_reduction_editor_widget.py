import os
import pytest


def test_cost_reduction_editor_set_get_roundtrip():
    # Force headless fallback implementation for reliable testing
    os.environ['DM_EDITOR_HEADLESS'] = '1'
    from dm_toolkit.gui.editor.forms.parts.cost_reduction_editor import CostReductionEditor

    w = CostReductionEditor()
    sample = [{"type": "PASSIVE", "amount": 2}, {"type": "ACTIVE_PAYMENT", "amount": 1}]
    w.set_value(sample)
    out = w.get_value()
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].get('type') == 'PASSIVE'
    # ensure ids were generated
    assert 'id' in out[0]
    assert 'id' in out[1]
