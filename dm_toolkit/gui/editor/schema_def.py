# -*- coding: utf-8 -*-
from enum import Enum, auto
from typing import Any, List, Optional, Dict, Union

class FieldType(Enum):
    """Enumeration of supported field types for the schema."""
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    SELECT = auto()       # Combo box from fixed options
    ZONE = auto()         # Zone selector
    FILTER = auto()       # Filter editor widget
    PLAYER = auto()       # Player scope selector
    LINK = auto()         # Variable Link widget
    GROUP = auto()        # Logical grouping (visual only)

class FieldSchema:
    """
    Defines the schema for a single UI field.
    """
    def __init__(
        self,
        key: str,
        label: str,
        field_type: FieldType,
        default: Any = None,
        options: Optional[List[Any]] = None, # For SELECT
        visible_if: Optional[Dict[str, Any]] = None, # Simple dependency: {other_key: value}
        tooltip: str = "",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        produces_output: bool = False # For LINK fields
    ):
        self.key = key
        self.label = label
        self.field_type = field_type
        self.default = default
        self.options = options or []
        self.visible_if = visible_if
        self.tooltip = tooltip
        self.min_value = min_value
        self.max_value = max_value
        self.produces_output = produces_output

class CommandSchema:
    """
    Defines the layout and fields for a specific Command Type.
    """
    def __init__(self, type_name: str, fields: List[FieldSchema], label_override: str = ""):
        self.type_name = type_name
        self.fields = fields
        self.label_override = label_override

# Example registry
SCHEMA_REGISTRY = {}

def register_schema(schema: CommandSchema):
    SCHEMA_REGISTRY[schema.type_name] = schema

def get_schema(type_name: str) -> Optional[CommandSchema]:
    return SCHEMA_REGISTRY.get(type_name)
