from typing import Any, Optional, List, Dict
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class HeadlessEditorItem(IEditorItem):
    def __init__(self, label: str = "", parent: Optional['HeadlessEditorItem'] = None):
        self._text = label
        self._parent = parent
        self._children: List['HeadlessEditorItem'] = []
        self._data: Dict[int, Any] = {}
        self._editable = True
        self._tool_tip = ""
        self._model: Optional['HeadlessEditorModel'] = None

    def row_count(self) -> int:
        return len(self._children)

    def child(self, row: int) -> Optional[IEditorItem]:
        if 0 <= row < len(self._children):
            return self._children[row]
        return None

    def append_row(self, item: IEditorItem) -> None:
        if isinstance(item, HeadlessEditorItem):
            item._parent = self
            item._model = self._model
            self._children.append(item)
        else:
            raise ValueError("Item must be HeadlessEditorItem")

    def insert_row(self, row: int, item: IEditorItem) -> None:
        if isinstance(item, HeadlessEditorItem):
            item._parent = self
            item._model = self._model
            self._children.insert(row, item)
        else:
            raise ValueError("Item must be HeadlessEditorItem")

    def remove_row(self, row: int) -> None:
        if 0 <= row < len(self._children):
            item = self._children.pop(row)
            item._parent = None

    def parent(self) -> Optional[IEditorItem]:
        return self._parent

    def data(self, role: int) -> Any:
        return self._data.get(role)

    def set_data(self, value: Any, role: int) -> None:
        self._data[role] = value

    def text(self) -> str:
        return self._text

    def set_text(self, text: str) -> None:
        self._text = text

    def is_editable(self) -> bool:
        return self._editable

    def set_editable(self, editable: bool) -> None:
        self._editable = editable

    def row(self) -> int:
        if self._parent:
            try:
                return self._parent._children.index(self)
            except ValueError:
                return -1
        return 0 # Root is row 0?

    def model(self) -> Optional['IEditorModel']:
        # If we have a stored model reference, use it.
        # Otherwise try to walk up to root and check if root has a model?
        if self._model:
            return self._model
        if self._parent:
            return self._parent.model()
        return None

    def set_tool_tip(self, tool_tip: str) -> None:
        self._tool_tip = tool_tip

class HeadlessEditorModel(IEditorModel):
    def __init__(self):
        self._root = HeadlessEditorItem("Root")
        self._root._model = self
        self._header_labels: List[str] = []

    def create_item(self, label: str) -> IEditorItem:
        return HeadlessEditorItem(label)

    def root_item(self) -> IEditorItem:
        return self._root

    def clear(self) -> None:
        self._root._children = []

    def append_row(self, item: IEditorItem) -> None:
        self._root.append_row(item)

    def item_from_index(self, index: Any) -> Optional[IEditorItem]:
        # Headless mode doesn't use QModelIndex logic usually.
        # But if we pass HeadlessEditorItem as index, we can return it.
        if isinstance(index, HeadlessEditorItem):
            return index
        return None

    def set_horizontal_header_labels(self, labels: List[str]) -> None:
        self._header_labels = labels
