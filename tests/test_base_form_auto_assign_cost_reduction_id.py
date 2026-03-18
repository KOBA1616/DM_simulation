import os

# Force dummy-Qt in tests
os.environ['DM_TOOLKIT_FORCE_DUMMY_QT'] = '1'

from PyQt6.QtWidgets import QApplication
from dm_toolkit.gui.editor.forms.base_form import BaseEditForm

# Ensure QApplication exists for real Qt environments
_app = QApplication.instance() or QApplication([])


class DummyWidget:
    def __init__(self):
        self.tooltip = None

    def setToolTip(self, t):
        self.tooltip = t


class DummyItem:
    def __init__(self, data):
        self._data = data
        self.set_called = False

    def data(self, role):
        return self._data

    def setData(self, data, role):
        self.set_called = True
        self._data = data

    def setText(self, text):
        pass


def test_auto_assign_cost_reduction_id_on_save():
    form = BaseEditForm()
    w = DummyWidget()
    form.bindings['dummy'] = w

    # cost_reductions missing id should be auto-assigned during save
    item = DummyItem({
        'type': 'CARD',
        'name': 'AutoIdCard',
        'cost_reductions': [{'type': 'PASSIVE', 'reduction_amount': 1}],
    })

    form.current_item = item
    form._is_populating = False

    form.save_data()

    assert item.set_called is True
    saved = item.data(None)
    assert 'cost_reductions' in saved
    cr0 = saved['cost_reductions'][0]
    assert 'id' in cr0 and isinstance(cr0['id'], str) and cr0['id'].startswith('auto_')
