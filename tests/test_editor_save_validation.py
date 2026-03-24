import os
os.environ['DM_TOOLKIT_FORCE_DUMMY_QT'] = '1'
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm


class DummyWidget:
    def __init__(self):
        self.style = None
        self.tooltip = None

    def setStyleSheet(self, s):
        self.style = s

    def setToolTip(self, t):
        self.tooltip = t


class DummyItem:
    def __init__(self, data):
        self._data = data
        self.set_called = False
        self.text = ""

    def data(self, role):
        return self._data

    def setData(self, data, role):
        self.set_called = True
        self._data = data

    def setText(self, text):
        self.text = text


def test_save_aborts_on_invalid_cost_reduction():
    form = BaseEditForm()
    # Bind a dummy widget so save_data will attempt to set style/tooltips
    w = DummyWidget()
    form.bindings['dummy'] = w

    # Prepare a current_item whose data contains invalid ACTIVE_PAYMENT
    item = DummyItem({
        'type': 'CARD',
        'name': 'X',
        'cost_reductions': [
            {'type': 'ACTIVE_PAYMENT'}  # missing required fields -> invalid
        ]
    })

    form.current_item = item
    form._is_populating = False

    # Attempt save; should abort and NOT call setData
    form.save_data()
    assert item.set_called is False
    # Widget should have been styled and given a tooltip
    assert w.style is not None and 'red' in w.style
    assert w.tooltip is not None and len(w.tooltip) > 0
