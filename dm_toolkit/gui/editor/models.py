# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional

class BaseModel:
    """Base wrapper for dictionary-based data models."""
    def __init__(self, data: Dict[str, Any] = None):
        self._data = data if data is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

class CommandModel(BaseModel):
    """Model wrapper for Action/Command data."""

    @property
    def type(self) -> str:
        return self.get('type', 'NONE')

    @type.setter
    def type(self, value: str):
        self.set('type', value)

    @property
    def format(self) -> str:
        return self.get('format', 'command')

    @format.setter
    def format(self, value: str):
        self.set('format', value)

    @property
    def amount(self) -> int:
        return int(self.get('amount', 0))

    @amount.setter
    def amount(self, value: int):
        self.set('amount', int(value))

    @property
    def target_filter(self) -> Dict[str, Any]:
        return self.get('target_filter', {})

    @target_filter.setter
    def target_filter(self, value: Dict[str, Any]):
        self.set('target_filter', value)

    @property
    def target_group(self) -> str:
        return self.get('target_group', 'PLAYER_SELF')

    @target_group.setter
    def target_group(self, value: str):
        self.set('target_group', value)


class CardModel(BaseModel):
    """Model wrapper for Card data."""
    pass

class CardNode(BaseModel):
    """
    Detailed validation model for Card Data.
    Used by DataManager to validate structure.
    """

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'CardNode':
        return cls(data)

    def validate(self) -> List[str]:
        """Performs validation on the card structure."""
        errors = []
        if not self.get('id'):
            errors.append("Missing ID")
        if not self.get('name'):
            errors.append("Missing Name")

        # Check effects structure
        effects = self.get('effects', []) or self.get('triggers', [])
        if not isinstance(effects, list):
            errors.append("Effects must be a list")

        return errors
