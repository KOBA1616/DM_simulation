import sys, types, importlib, importlib.machinery
m = types.ModuleType('PyQt6')
m.__spec__ = importlib.machinery.ModuleSpec('PyQt6', None)
# Minimal submodules expected by tests
class MockModule(types.SimpleNamespace):
    pass
sys.modules['PyQt6'] = m
sys.modules['PyQt6.QtWidgets'] = MockModule()
sys.modules['PyQt6.QtGui'] = MockModule()
sys.modules['PyQt6.QtCore'] = MockModule()
# Provide minimal classes/attributes expected by tests
class _QStandardItem:
    def __init__(self, *args, **kwargs):
        self._children = []
        self._data = {}
    def data(self, role=None):
        return self._data.get(role)
    def setData(self, value, role=None):
        self._data[role] = value
    def appendRow(self, item):
        self._children.append(item)
    def rowCount(self):
        return len(self._children)
    def child(self, idx):
        return self._children[idx] if 0 <= idx < len(self._children) else None

class _QStandardItemModel:
    def __init__(self):
        self._root = _QStandardItem('root')
    def invisibleRootItem(self):
        return self._root
    def clear(self):
        self._root = _QStandardItem('root')
    def appendRow(self, item):
        self._root.appendRow(item)

class _Qt:
    class ItemDataRole:
        UserRole = 256
        DisplayRole = 0

sys.modules['PyQt6.QtGui'].QStandardItem = _QStandardItem
sys.modules['PyQt6.QtGui'].QStandardItemModel = _QStandardItemModel
sys.modules['PyQt6.QtCore'].Qt = _Qt

import pytest
args = sys.argv[1:] if len(sys.argv) > 1 else ['-q', 'tests/gui', '-q']
sys.exit(pytest.main(args))
