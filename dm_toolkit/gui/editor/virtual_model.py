# -*- coding: utf-8 -*-
from typing import List, Optional, Any, Dict

class VirtualStandardItem:
    """
    A lightweight, pure-Python implementation of QStandardItem for headless environments.
    Mimics the API used by the Editor logic.
    """
    def __init__(self, text: str = ""):
        self._text = text
        self._data: Dict[int, Any] = {}
        self._children: List['VirtualStandardItem'] = []
        self._parent: Optional['VirtualStandardItem'] = None
        self._model: Optional['VirtualStandardItemModel'] = None

    def text(self) -> str:
        return self._text

    def setText(self, text: str):
        self._text = text

    def data(self, role: int = 257) -> Any:
        # Default role often matches Qt.UserRole+1 if not specified,
        # but explicit calls usually provide role.
        return self._data.get(role)

    def setData(self, value: Any, role: int = 257):
        self._data[role] = value

    def appendRow(self, item: 'VirtualStandardItem'):
        if item._parent:
             # Reparenting logic would be needed here if strictly emulating Qt,
             # but for our usage, items usually new.
             pass
        item._parent = self
        item._model = self._model
        self._children.append(item)

    def removeRow(self, row: int):
        if 0 <= row < len(self._children):
            item = self._children.pop(row)
            item._parent = None
            item._model = None

    def child(self, row: int, column: int = 0) -> Optional['VirtualStandardItem']:
        # We assume column 0 for list-like tree
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def rowCount(self) -> int:
        return len(self._children)

    def parent(self) -> Optional['VirtualStandardItem']:
        return self._parent

    def model(self) -> Optional['VirtualStandardItemModel']:
        return self._model

    def index(self):
        # In virtual mode, the item itself can act as the index
        # or we return a lightweight wrapper.
        # For simplicity, let's return self, and ensure model.itemFromIndex handles it.
        return self

    def row(self) -> int:
        if self._parent:
            try:
                return self._parent._children.index(self)
            except ValueError:
                return -1
        return 0

    # Stubs for compatibility
    def setEditable(self, editable: bool):
        pass

    def setToolTip(self, tooltip: str):
        pass


class VirtualStandardItemModel:
    """
    A lightweight, pure-Python implementation of QStandardItemModel.
    """
    def __init__(self):
        self._root = VirtualStandardItem()
        self._root._model = self

    def invisibleRootItem(self) -> VirtualStandardItem:
        return self._root

    def appendRow(self, item: VirtualStandardItem):
        self._root.appendRow(item)

    def clear(self):
        self._root._children = []

    def itemFromIndex(self, index: Any) -> Optional[VirtualStandardItem]:
        # Since VirtualStandardItem.index() returns self, we just return it.
        if isinstance(index, VirtualStandardItem):
            return index
        return None

    def setHorizontalHeaderLabels(self, labels: List[str]):
        pass
