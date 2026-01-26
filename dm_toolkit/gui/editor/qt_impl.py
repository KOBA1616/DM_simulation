# -*- coding: utf-8 -*-
from typing import Any, List, Optional
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import QModelIndex

from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class QtEditorItem(IEditorItem):
    def __init__(self, item: QStandardItem):
        self.item = item

    def rowCount(self) -> int:
        return self.item.rowCount()

    def row(self) -> int:
        return self.item.row()

    def child(self, row: int) -> Optional[IEditorItem]:
        c = self.item.child(row)
        return QtEditorItem(c) if c else None

    def appendRow(self, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self.item.appendRow(item.item)
        else:
            raise TypeError("Expected QtEditorItem")

    def removeRow(self, row: int) -> None:
        self.item.removeRow(row)

    def data(self, role: int) -> Any:
        return self.item.data(role)

    def setData(self, value: Any, role: int) -> None:
        self.item.setData(value, role)

    def parent(self) -> Optional[IEditorItem]:
        p = self.item.parent()
        return QtEditorItem(p) if p else None

    def text(self) -> str:
        return self.item.text()

    def setText(self, text: str) -> None:
        self.item.setText(text)

    def setEditable(self, editable: bool) -> None:
        self.item.setEditable(editable)

    def setToolTip(self, toolTip: str) -> None:
        self.item.setToolTip(toolTip)

class QtEditorModel(IEditorModel):
    def __init__(self, model: QStandardItemModel):
        self.model = model

    def create_item(self, label: str) -> IEditorItem:
        return QtEditorItem(QStandardItem(label))

    def invisibleRootItem(self) -> IEditorItem:
        return QtEditorItem(self.model.invisibleRootItem())

    def appendRow(self, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self.model.appendRow(item.item)
        else:
            raise TypeError("Expected QtEditorItem")

    def clear(self) -> None:
        self.model.clear()

    def setHorizontalHeaderLabels(self, labels: List[str]) -> None:
        self.model.setHorizontalHeaderLabels(labels)

    def itemFromIndex(self, index: Any) -> Optional[IEditorItem]:
        if isinstance(index, QModelIndex):
            item = self.model.itemFromIndex(index)
            return QtEditorItem(item) if item else None
        return None
