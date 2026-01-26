from abc import ABC, abstractmethod
from typing import Any, Optional, List

class IEditorItem(ABC):
    @abstractmethod
    def row_count(self) -> int:
        pass

    @abstractmethod
    def child(self, row: int) -> Optional['IEditorItem']:
        pass

    @abstractmethod
    def append_row(self, item: 'IEditorItem') -> None:
        pass

    @abstractmethod
    def insert_row(self, row: int, item: 'IEditorItem') -> None:
        pass

    @abstractmethod
    def remove_row(self, row: int) -> None:
        pass

    @abstractmethod
    def parent(self) -> Optional['IEditorItem']:
        pass

    @abstractmethod
    def data(self, role: int) -> Any:
        pass

    @abstractmethod
    def set_data(self, value: Any, role: int) -> None:
        pass

    @abstractmethod
    def text(self) -> str:
        pass

    @abstractmethod
    def set_text(self, text: str) -> None:
        pass

    @abstractmethod
    def is_editable(self) -> bool:
        pass

    @abstractmethod
    def set_editable(self, editable: bool) -> None:
        pass

    @abstractmethod
    def row(self) -> int:
        pass

    @abstractmethod
    def model(self) -> Optional['IEditorModel']:
        pass

    @abstractmethod
    def set_tool_tip(self, tool_tip: str) -> None:
        pass

class IEditorModel(ABC):
    @abstractmethod
    def create_item(self, label: str) -> IEditorItem:
        pass

    @abstractmethod
    def root_item(self) -> IEditorItem:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def append_row(self, item: IEditorItem) -> None:
        pass

    @abstractmethod
    def item_from_index(self, index: Any) -> Optional[IEditorItem]:
        pass

    @abstractmethod
    def set_horizontal_header_labels(self, labels: List[str]) -> None:
        pass
