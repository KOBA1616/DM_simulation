# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Any
from PyQt6.QtCore import QObject

# QObject has a specific metaclass (sip.wrappertype), and ABC has ABCMeta.
# To mix them, we need to handle the metaclass conflict.
# Usually, inheriting from QObject first is enough if the interface doesn't use ABCMeta strictly or we define a combined metaclass.
# For simplicity in PyQt, we can make the interface just a Mixin without strict ABCMeta if feasible, or use a combined metaclass.

class EditorWidgetInterface(ABC):
    @abstractmethod
    def get_value(self) -> Any:
        pass

    @abstractmethod
    def set_value(self, value: Any) -> None:
        pass
