# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

class IEditorItem(ABC):
    """Abstract interface for tree items in the editor."""

    @abstractmethod
    def rowCount(self) -> int: ...

    @abstractmethod
    def row(self) -> int: ...

    @abstractmethod
    def child(self, row: int) -> Optional['IEditorItem']: ...

    @abstractmethod
    def appendRow(self, item: 'IEditorItem') -> None: ...

    @abstractmethod
    def removeRow(self, row: int) -> None: ...

    @abstractmethod
    def data(self, role: int) -> Any: ...

    @abstractmethod
    def setData(self, value: Any, role: int) -> None: ...

    @abstractmethod
    def parent(self) -> Optional['IEditorItem']: ...

    @abstractmethod
    def text(self) -> str: ...

    @abstractmethod
    def setText(self, text: str) -> None: ...

    @abstractmethod
    def setEditable(self, editable: bool) -> None: ...

    @abstractmethod
    def setToolTip(self, toolTip: str) -> None: ...

class IEditorModel(ABC):
    """Abstract interface for the tree model."""

    @abstractmethod
    def create_item(self, label: str) -> IEditorItem: ...

    @abstractmethod
    def invisibleRootItem(self) -> IEditorItem: ...

    @abstractmethod
    def appendRow(self, item: IEditorItem) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def setHorizontalHeaderLabels(self, labels: List[str]) -> None: ...

    @abstractmethod
    def itemFromIndex(self, index: Any) -> Optional[IEditorItem]:
        """Convert an opaque index (e.g. QModelIndex) to an item.
        If index is already an item or not applicable, handle gracefully.
        """
        ...
