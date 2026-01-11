# -*- coding: utf-8 -*-
from typing import Optional, Dict, List, Any, Type, TypeVar
from abc import ABC, abstractmethod
import uuid
from dm_toolkit.gui.editor.models import CommandModel, EffectModel, CardModel, ModifierModel, ReactionModel
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseNode(ABC):
    """
    Abstract base class for DataNodes.
    Wraps a Pydantic model and handles data access, serving as the primary API for UI components.
    This enforces the 'Data Structure Unification' by hiding the raw dictionary/Qt Item implementation details.
    """
    def __init__(self, data: Dict[str, Any], model_class: Type[T]):
        self._raw_data = data
        self._model_class = model_class
        self._ensure_uid()
        # We hold the raw dict reference (from QStandardItem) to allow in-place updates

    def _ensure_uid(self):
        if 'uid' not in self._raw_data:
            self._raw_data['uid'] = str(uuid.uuid4())

    @property
    def uid(self):
        return self._raw_data.get('uid')

    def to_dict(self) -> Dict[str, Any]:
        """Returns the raw dictionary for serialization."""
        return self._raw_data

    def to_model(self) -> T:
        """Validates and returns the Pydantic model instance."""
        # Handle strict validation vs permissive loading?
        # For Editor, we usually want permissive, but to_model implies validity.
        try:
            return self._model_class(**self._raw_data)
        except Exception:
            # Fallback construct
            return self._model_class.construct(**self._raw_data)

    def update(self, new_data: Dict[str, Any]):
        """Updates the internal data with a new dictionary."""
        self._raw_data.update(new_data)

    def set_field(self, key: str, value: Any):
        """Sets a specific field, updating the raw data."""
        self._raw_data[key] = value

    def get_field(self, key: str, default: Any = None) -> Any:
        return self._raw_data.get(key, default)

class CommandNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, CommandModel)

    @property
    def type(self) -> str:
        return self._raw_data.get('type', 'NONE')

class EffectNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, EffectModel)

    @property
    def trigger(self) -> str:
        return self._raw_data.get('trigger', 'NONE')

class CardNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, CardModel)

    @property
    def name(self):
        return self._raw_data.get('name', 'No Name')

    @property
    def id(self):
        return self._raw_data.get('id', 0)

def create_node(role_type: str, data: Dict[str, Any]) -> BaseNode:
    if role_type == "CARD":
        return CardNode(data)
    elif role_type == "EFFECT":
        return EffectNode(data)
    elif role_type == "COMMAND":
        return CommandNode(data)
    elif role_type == "MODIFIER":
        return BaseNode(data, ModifierModel)
    elif role_type == "REACTION_ABILITY":
        return BaseNode(data, ReactionModel)
    # Fallback
    return BaseNode(data, BaseModel)
