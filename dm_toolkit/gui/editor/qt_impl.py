from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtCore import QModelIndex
from typing import Any, Optional, List
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class QtEditorItem(IEditorItem):
    def __init__(self, item: QStandardItem):
        self._item = item

    def row_count(self) -> int:
        return self._item.rowCount()

    def child(self, row: int) -> Optional[IEditorItem]:
        child = self._item.child(row)
        return QtEditorItem(child) if child else None

    def append_row(self, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self._item.appendRow(item._item)
        else:
            raise ValueError("Item must be QtEditorItem")

    def insert_row(self, row: int, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self._item.insertRow(row, item._item)
        else:
            raise ValueError("Item must be QtEditorItem")

    def remove_row(self, row: int) -> None:
        self._item.removeRow(row)

    def parent(self) -> Optional[IEditorItem]:
        p = self._item.parent()
        return QtEditorItem(p) if p else None

    def data(self, role: int) -> Any:
        return self._item.data(role)

    def set_data(self, value: Any, role: int) -> None:
        self._item.setData(value, role)

    def text(self) -> str:
        return self._item.text()

    def set_text(self, text: str) -> None:
        self._item.setText(text)

    def is_editable(self) -> bool:
        return self._item.isEditable()

    def set_editable(self, editable: bool) -> None:
        self._item.setEditable(editable)

    def row(self) -> int:
        return self._item.row()

    def model(self) -> Optional['IEditorModel']:
        m = self._item.model()
        return QtEditorModel(m) if m else None

    def set_tool_tip(self, tool_tip: str) -> None:
        self._item.setToolTip(tool_tip)

    def __eq__(self, other):
        if isinstance(other, QtEditorItem):
            return self._item == other._item
        return NotImplemented

    def __hash__(self):
        return hash(self._item)

    # Helper to get raw item
    def get_raw_item(self) -> QStandardItem:
        return self._item

class QtEditorModel(IEditorModel):
    def __init__(self, model: QStandardItemModel):
        self._model = model

    def create_item(self, label: str) -> IEditorItem:
        return QtEditorItem(QStandardItem(label))

    def root_item(self) -> IEditorItem:
        return QtEditorItem(self._model.invisibleRootItem())

    def clear(self) -> None:
        self._model.clear()

    def append_row(self, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self._model.appendRow(item._item)
        else:
            raise ValueError("Item must be QtEditorItem")

    def item_from_index(self, index: Any) -> Optional[IEditorItem]:
        if isinstance(index, QModelIndex):
            item = self._model.itemFromIndex(index)
            return QtEditorItem(item) if item else None
        return None

    def set_horizontal_header_labels(self, labels: List[str]) -> None:
        self._model.setHorizontalHeaderLabels(labels)

    # Helper to get raw model
    def get_raw_model(self) -> QStandardItemModel:
        return self._model
