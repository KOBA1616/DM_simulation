# dm_toolkit/editor/core/headless_impl.py
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel
from typing import Any, Optional, List, Dict

class HeadlessEditorItem(IEditorItem):
    def __init__(self, label: str = ""):
        self._data: Dict[int, Any] = {}
        self._children: List['HeadlessEditorItem'] = []
        self._parent: Optional['HeadlessEditorItem'] = None
        self._text = label
        self._editable = True

    def data(self, role: int) -> Any:
        return self._data.get(role)

    def set_data(self, value: Any, role: int) -> None:
        self._data[role] = value

    def parent(self) -> Optional['IEditorItem']:
        return self._parent

    def child(self, row: int) -> Optional['IEditorItem']:
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def row_count(self) -> int:
        return len(self._children)

    def append_row(self, item: 'IEditorItem') -> None:
        if isinstance(item, HeadlessEditorItem):
            item._parent = self
            self._children.append(item)
        else:
            raise ValueError(f"Expected HeadlessEditorItem, got {type(item)}")

    def row(self) -> int:
        if self._parent:
            try:
                return self._parent._children.index(self)
            except ValueError:
                return -1
        return -1

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def set_editable(self, editable: bool) -> None:
        self._editable = editable

    def get_id(self) -> Any:
        return self


class HeadlessEditorModel(IEditorModel):
    def __init__(self):
        self._root = HeadlessEditorItem("ROOT")
        self._headers: List[str] = []

    def create_item(self, label: str) -> IEditorItem:
        return HeadlessEditorItem(label)

    def root_item(self) -> IEditorItem:
        return self._root

    def get_item(self, handle: Any) -> Optional[IEditorItem]:
        if isinstance(handle, HeadlessEditorItem):
            return handle
        if handle is None:
             return None # Or root? Usually None handle means failure or root depending on context.
             # In Qt, QModelIndex() is root parent. But here handle IS the item.
        return None

    def remove_row(self, row: int, parent_handle: Any) -> bool:
        parent = self.get_item(parent_handle)
        if not parent:
            # If handle is None, maybe we meant root?
            # In removeRow(row, parent), if parent is invalid index, it means root.
            if parent_handle is None: # Assuming None handle => Root
                 parent = self._root
            else:
                 return False

        if isinstance(parent, HeadlessEditorItem):
            if 0 <= row < len(parent._children):
                del parent._children[row]
                return True
        return False

    def clear(self) -> None:
        self._root = HeadlessEditorItem("ROOT")

    def set_headers(self, labels: List[str]) -> None:
        self._headers = labels

    def append_row(self, item: IEditorItem) -> None:
        self._root.append_row(item)
