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


def test_save_proceeds_with_conflict_warning():
    form = BaseEditForm()
    w = DummyWidget()
    form.bindings['dummy'] = w

    item = DummyItem({
        'type': 'CARD',
        'name': 'ConflictCard',
        'cost_reductions': [{'type': 'PASSIVE'}],
        'static_abilities': [{'type': 'COST_MODIFIER', 'value': 1}]
    })

    form.current_item = item
    form._is_populating = False

    form.save_data()
    # Save should proceed (setData called)
    assert item.set_called is True
    # Widget should have a tooltip warning
    assert w.tooltip is not None and 'PASSIVE' in w.tooltip
