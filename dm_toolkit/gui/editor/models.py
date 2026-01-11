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

    @property
    def str_param(self) -> str:
        return self.get('str_param', '')

    @str_param.setter
    def str_param(self, value: str):
        self.set('str_param', value)

    @property
    def mutation_kind(self) -> str:
        return self.get('mutation_kind', '')

    @mutation_kind.setter
    def mutation_kind(self, value: str):
        self.set('mutation_kind', value)

    @property
    def from_zone(self) -> str:
        return self.get('from_zone', 'NONE')

    @from_zone.setter
    def from_zone(self, value: str):
        self.set('from_zone', value)

    @property
    def to_zone(self) -> str:
        return self.get('to_zone', 'NONE')

    @to_zone.setter
    def to_zone(self, value: str):
        self.set('to_zone', value)

    @property
    def optional(self) -> bool:
        return bool(self.get('optional', False))

    @optional.setter
    def optional(self, value: bool):
        self.set('optional', value)

    @property
    def play_flags(self) -> List[str]:
        return self.get('play_flags', [])

    @play_flags.setter
    def play_flags(self, value: List[str]):
        self.set('play_flags', value)

    @property
    def flags(self) -> List[str]:
        return self.get('flags', [])

    @flags.setter
    def flags(self, value: List[str]):
        self.set('flags', value)

    @property
    def ref_mode(self) -> str:
        return self.get('ref_mode', 'NONE')

    @ref_mode.setter
    def ref_mode(self, value: str):
        self.set('ref_mode', value)

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
