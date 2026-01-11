# -*- coding: utf-8 -*-
from typing import Optional, Dict, List, Any
from abc import ABC, abstractmethod
import uuid
from dm_toolkit.gui.editor.schema import models as m

class BaseNode(ABC):
    """
    Abstract base class for DataNodes.
    Wraps a Pydantic model and handles data access.
    """
    def __init__(self, data: Dict[str, Any], model_class):
        self.raw_data = data
        self.model_class = model_class
        self._ensure_uid()
        # We validate on init but keep working on raw_data to allow partial/invalid states in UI
        # self.validate() # Optional: strict validation on load?

    def _ensure_uid(self):
        if 'uid' not in self.raw_data:
            self.raw_data['uid'] = str(uuid.uuid4())

    @property
    def uid(self):
        return self.raw_data.get('uid')

    def to_dict(self) -> Dict[str, Any]:
        """Returns the raw dictionary for serialization."""
        return self.raw_data

    def validate(self):
        """Validates the current data against the Pydantic model."""
        return self.model_class(**self.raw_data)

    def update(self, new_data: Dict[str, Any]):
        """Updates the internal data with a new dictionary."""
        self.raw_data.update(new_data)

class CommandNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, m.CommandDef)

    def get_type(self) -> str:
        return self.raw_data.get('type', 'NONE')

class EffectNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, m.EffectDef)

    def get_trigger(self) -> str:
        return self.raw_data.get('trigger', 'NONE')

class CardNode(BaseNode):
    def __init__(self, data: Dict[str, Any]):
        super().__init__(data, m.CardData)

    @property
    def name(self):
        return self.raw_data.get('name', 'No Name')

    @property
    def id(self):
        return self.raw_data.get('id', 0)

def create_node(role_type: str, data: Dict[str, Any]) -> BaseNode:
    if role_type == "CARD":
        return CardNode(data)
    elif role_type == "EFFECT":
        return EffectNode(data)
    elif role_type == "COMMAND":
        return CommandNode(data)
    # Fallback
    return BaseNode(data, None) # Generic wrapper
