# -*- coding: utf-8 -*-
from typing import Any, List, Optional, Dict
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class HeadlessEditorItem(IEditorItem):
    def __init__(self, text: str = "", parent: Optional['HeadlessEditorItem'] = None):
        self._text = text
        self._parent = parent
        self._children: List[HeadlessEditorItem] = []
        self._data: Dict[int, Any] = {}
        self._editable = True
        self._tool_tip = ""

    def rowCount(self) -> int:
        return len(self._children)

    def row(self) -> int:
        if self._parent:
            try:
                return self._parent._children.index(self)
            except ValueError:
                return -1
        return 0

    def child(self, row: int) -> Optional[IEditorItem]:
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def appendRow(self, item: IEditorItem) -> None:
        if isinstance(item, HeadlessEditorItem):
            item._parent = self
            self._children.append(item)
        else:
            raise TypeError(f"Expected HeadlessEditorItem, got {type(item)}")

    def removeRow(self, row: int) -> None:
        if 0 <= row < len(self._children):
            self._children.pop(row)

    def data(self, role: int) -> Any:
        return self._data.get(role)

    def setData(self, value: Any, role: int) -> None:
        self._data[role] = value

    def parent(self) -> Optional[IEditorItem]:
        return self._parent

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text

    def setEditable(self, editable: bool) -> None:
        self._editable = editable

    def setToolTip(self, toolTip: str) -> None:
        self._tool_tip = toolTip

class HeadlessEditorModel(IEditorModel):
    def __init__(self):
        self._root = HeadlessEditorItem("Root")

    def create_item(self, label: str) -> IEditorItem:
        return HeadlessEditorItem(label)

    def invisibleRootItem(self) -> IEditorItem:
        return self._root

    def appendRow(self, item: IEditorItem) -> None:
        self._root.appendRow(item)

    def clear(self) -> None:
        self._root = HeadlessEditorItem("Root")

    def setHorizontalHeaderLabels(self, labels: List[str]) -> None:
        pass # No-op for headless

    def itemFromIndex(self, index: Any) -> Optional[IEditorItem]:
        # Headless mode likely won't use QModelIndex logic.
        # We can implement if needed, e.g. if index IS an item.
        if isinstance(index, HeadlessEditorItem):
            return index
        return None
