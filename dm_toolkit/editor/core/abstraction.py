# dm_toolkit/editor/core/abstraction.py
from abc import ABC, abstractmethod
from typing import Any, Optional, List

class IEditorItem(ABC):
    @abstractmethod
    def data(self, role: int) -> Any:
        pass

    @abstractmethod
    def set_data(self, value: Any, role: int) -> None:
        pass

    @abstractmethod
    def parent(self) -> Optional['IEditorItem']:
        pass

    @abstractmethod
    def child(self, row: int) -> Optional['IEditorItem']:
        pass

    @abstractmethod
    def row_count(self) -> int:
        pass

    @abstractmethod
    def append_row(self, item: 'IEditorItem') -> None:
        pass

    @abstractmethod
    def row(self) -> int:
        pass

    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def set_text(self, text: str) -> None:
        pass

    @abstractmethod
    def set_editable(self, editable: bool) -> None:
        pass

    @abstractmethod
    def get_id(self) -> Any:
        """Returns a unique identifier or handle for this item."""
        pass


class IEditorModel(ABC):
    @abstractmethod
    def create_item(self, label: str) -> IEditorItem:
        pass

    @abstractmethod
    def root_item(self) -> IEditorItem:
        pass

    @abstractmethod
    def get_item(self, handle: Any) -> Optional[IEditorItem]:
        """
        Retrieve an item based on a handle (e.g., QModelIndex, ID, or the item itself).
        """
        pass

    @abstractmethod
    def remove_row(self, row: int, parent_handle: Any) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def set_headers(self, labels: List[str]) -> None:
        pass

    @abstractmethod
    def append_row(self, item: IEditorItem) -> None:
        """Appends a row to the root item."""
        pass
