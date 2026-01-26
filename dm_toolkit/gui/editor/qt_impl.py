# dm_toolkit/gui/editor/qt_impl.py
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtCore import QModelIndex
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel
from typing import Any, Optional, List

class QtEditorItem(IEditorItem):
    def __init__(self, item: QStandardItem):
        self._item = item

    def data(self, role: int) -> Any:
        return self._item.data(role)

    def set_data(self, value: Any, role: int) -> None:
        self._item.setData(value, role)

    def parent(self) -> Optional['IEditorItem']:
        p = self._item.parent()
        if p:
            return QtEditorItem(p)
        return None

    def child(self, row: int) -> Optional['IEditorItem']:
        c = self._item.child(row)
        if c:
            return QtEditorItem(c)
        return None

    def row_count(self) -> int:
        return self._item.rowCount()

    def append_row(self, item: 'IEditorItem') -> None:
        if isinstance(item, QtEditorItem):
            self._item.appendRow(item._item)
        else:
            raise ValueError(f"Expected QtEditorItem, got {type(item)}")

    def row(self) -> int:
        return self._item.row()

    def text(self) -> str:
        return self._item.text()

    def set_text(self, text: str) -> None:
        self._item.setText(text)

    def set_editable(self, editable: bool) -> None:
        self._item.setEditable(editable)

    def get_id(self) -> Any:
        return self._item

class QtEditorModel(IEditorModel):
    def __init__(self, model: QStandardItemModel):
        self._model = model

    def create_item(self, label: str) -> IEditorItem:
        return QtEditorItem(QStandardItem(label))

    def root_item(self) -> IEditorItem:
        return QtEditorItem(self._model.invisibleRootItem())

    def get_item(self, handle: Any) -> Optional[IEditorItem]:
        if isinstance(handle, QModelIndex):
            if handle.isValid():
                return QtEditorItem(self._model.itemFromIndex(handle))
            else:
                # Typically QModelIndex() (invalid) represents the root parent in some contexts,
                # but itemFromIndex(invalid) returns None.
                # If the caller expects an item for an invalid index, it usually means they made a mistake
                # or we should handle it. LogicTreeWidget checks isValid() before calling.
                return None
        elif isinstance(handle, QtEditorItem):
            return handle
        elif isinstance(handle, QStandardItem):
            return QtEditorItem(handle)
        return None

    def remove_row(self, row: int, parent_handle: Any) -> bool:
        parent_index = QModelIndex()
        if isinstance(parent_handle, QModelIndex):
            parent_index = parent_handle
        elif isinstance(parent_handle, QtEditorItem):
            parent_index = parent_handle._item.index()
        elif isinstance(parent_handle, QStandardItem):
            parent_index = parent_handle.index()

        return self._model.removeRow(row, parent_index)

    def clear(self) -> None:
        self._model.clear()

    def set_headers(self, labels: List[str]) -> None:
        self._model.setHorizontalHeaderLabels(labels)

    def append_row(self, item: IEditorItem) -> None:
        if isinstance(item, QtEditorItem):
            self._model.appendRow(item._item)
        else:
            raise ValueError(f"Expected QtEditorItem, got {type(item)}")
