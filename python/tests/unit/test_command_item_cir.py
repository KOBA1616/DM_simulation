from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.models import CommandModel


class SimpleItem:
    def __init__(self, label=''):
        self._data = {}
        self._children = []
        self.label = label

    def set_data(self, value, role):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def append_row(self, item):
        self._children.append(item)

    def row_count(self):
        return len(self._children)

    def child(self, idx):
        return self._children[idx]


class SimpleModel:
    def create_item(self, label=''):
        return SimpleItem(label)


def test_create_command_item_attaches_cir():
    ser = ModelSerializer()
    cmd = CommandModel.construct(type='TEST', params={}, if_true=[], if_false=[], options=[])
    cmd._cir = [{'kind': 'COMMAND', 'type': 'X'}]

    model = SimpleModel()
    item = ser.create_command_item(model, cmd)

    assert item.data('ROLE_CIR') == [{'kind': 'COMMAND', 'type': 'X'}]
