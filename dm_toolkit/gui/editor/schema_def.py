# -*- coding: utf-8 -*-
# schema_def.py
# UIスキーマ定義。unified_action_form.py が UI を動的に生成するために使用します。
# 維持: AIが編集する際は unified_action_form.py や ACTION_UI_CONFIG (command_config.py) とセットで理解する必要があります。
# 重要: このファイルは削除しないでください。unified_action_form.py の動作に不可欠です。

from enum import Enum, auto
from typing import Any, List, Optional, Dict, Union
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader

class FieldType(Enum):
    """Enumeration of supported field types for the schema."""
    INT = auto()          # Spinbox
    FLOAT = auto()        # DoubleSpinbox
    STRING = auto()       # Text Line Edit
    BOOL = auto()         # Checkbox
    SELECT = auto()       # Combo box from fixed options
    ZONE = auto()         # Zone selector
    FILTER = auto()       # Filter editor widget
    PLAYER = auto()       # Player scope selector
    LINK = auto()         # Variable Link widget
    GROUP = auto()        # Logical grouping (visual only) - Not fully implemented yet
    OPTIONS_CONTROL = auto() # Special control for generating option branches
    CIVILIZATION = auto() # Civilization selector
    TYPE_SELECT = auto()  # Card type selector
    RACES = auto()        # Races editor
    ENUM = auto()         # Dynamically loaded Enum
    CONDITION = auto()    # Condition editor
    CONDITION_TREE = auto() # Hierarchical condition tree editor

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
        produces_output: bool = False, # For LINK fields
        widget_hint: Optional[str] = None, # Hint for factory (e.g. 'ref_mode_combo')
        enum_source: Optional[str] = None, # For ENUM type: 'dm_ai_module.ClassName'
        step: Optional[int] = None # For INT fields: setSingleStep(step)
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
        self.widget_hint = widget_hint
        self.enum_source = enum_source
        self.step = step

class CommandSchema:
    """
    Defines the layout and fields for a specific Command Type.
    """
    def __init__(self, type_name: str, fields: List[FieldSchema], label_override: str = "", allowed_filter_fields: Optional[List[str]] = None, default_amount: Optional[int] = None, targeting_mode: Optional[str] = None):
        self.type_name = type_name
        self.fields = fields
        self.label_override = label_override
        # Optional constraint hint for Filter widgets (e.g., ['zones','types'])
        self.allowed_filter_fields = allowed_filter_fields
        self.default_amount = default_amount
        self.targeting_mode = targeting_mode

# Registry
SCHEMA_REGISTRY = {}
COMMAND_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _build_group_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    try:
        groups = EditorConfigLoader.get_command_groups() or {}
        for gname, cmds in groups.items():
            if not isinstance(cmds, list):
                continue
            for cmd in cmds:
                if isinstance(cmd, str):
                    lookup[cmd] = str(gname)
    except Exception:
        return {}
    return lookup


def _resolve_command_group(type_name: str) -> str:
    try:
        cfg = EditorConfigLoader.get_command_ui_config().get(type_name, {})
        if isinstance(cfg, dict):
            explicit = cfg.get('group')
            if isinstance(explicit, str) and explicit:
                return explicit
    except Exception:
        pass
    lookup = _build_group_lookup()
    return lookup.get(type_name, 'OTHER')


def _resolve_validator_hint(type_name: str) -> str:
    try:
        cfg = EditorConfigLoader.get_command_ui_config().get(type_name, {})
        if isinstance(cfg, dict):
            v = cfg.get('validator')
            if isinstance(v, str):
                return v
    except Exception:
        pass
    return ''


def _resolve_text_hint(type_name: str) -> str:
    try:
        cfg = EditorConfigLoader.get_command_ui_config().get(type_name, {})
        if isinstance(cfg, dict):
            hint = cfg.get('text_hint') or cfg.get('description')
            if isinstance(hint, str):
                return hint
    except Exception:
        pass
    return ''


def _sync_command_registry_entry(schema: 'CommandSchema') -> None:
    """Keep command metadata SSOT synchronized with schema registration.

    Required spec fields: type, group, fields, validator, text_hint.
    """
    COMMAND_REGISTRY[schema.type_name] = {
        'type': schema.type_name,
        'group': _resolve_command_group(schema.type_name),
        'fields': [f.key for f in schema.fields],
        'validator': _resolve_validator_hint(schema.type_name),
        'text_hint': _resolve_text_hint(schema.type_name),
    }


def get_registered_command_types() -> List[str]:
    if COMMAND_REGISTRY:
        return sorted(COMMAND_REGISTRY.keys())
    return sorted(SCHEMA_REGISTRY.keys())


def get_command_registry_entry(type_name: str) -> Optional[Dict[str, Any]]:
    return COMMAND_REGISTRY.get(type_name)


def get_command_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    return dict(COMMAND_REGISTRY)

def register_schema(schema: CommandSchema):
    SCHEMA_REGISTRY[schema.type_name] = schema
    _sync_command_registry_entry(schema)

def get_schema(type_name: str) -> Optional[CommandSchema]:
    return SCHEMA_REGISTRY.get(type_name)

class SchemaLoader:
    """
    Loads raw configuration from EditorConfigLoader and populates the SCHEMA_REGISTRY.
    Centralizes the logic that was previously hardcoded in Forms.
    """

    # Default mapping for field names to types/widgets
    _DEFAULT_MAPPING = {
        'target_group':  {'type': FieldType.PLAYER, 'label': 'Target Player', 'hint': 'player_scope'},
        'target_filter': {'type': FieldType.FILTER, 'label': 'Filter'},
        'amount':        {'type': FieldType.INT, 'label': 'Amount', 'min': 1},
        'str_param':     {'type': FieldType.STRING, 'label': 'String Param'},
        'str_val':       {'type': FieldType.STRING, 'label': 'String Value'},
        'from_zone':     {'type': FieldType.ZONE, 'label': 'From Zone', 'hint': 'zone_combo'},
        'to_zone':       {'type': FieldType.ZONE, 'label': 'To Zone', 'hint': 'zone_combo'},
        'optional':      {'type': FieldType.BOOL, 'label': 'Optional'},
        'up_to':         {'type': FieldType.BOOL, 'label': 'Up To'},
        'input_link':    {'type': FieldType.LINK, 'label': 'Input Variables'},
        'output_link':   {'type': FieldType.LINK, 'label': 'Output Variables', 'produces_output': True},
        'mutation_kind': {'type': FieldType.STRING, 'label': 'Mutation Kind'},
        'ref_mode':      {'type': FieldType.SELECT, 'label': 'Reference Mode', 'hint': 'ref_mode_combo'},
        'generate_opts': {'type': FieldType.OPTIONS_CONTROL, 'label': 'Options', 'hint': 'options_control'},
        'condition':     {'type': FieldType.CONDITION, 'label': 'Condition'}
    }

    @classmethod
    def load_schemas(cls):
        """Parses the loaded COMMAND_UI_CONFIG and registers schemas."""
        if SCHEMA_REGISTRY:
            return

        raw_config = EditorConfigLoader.get_command_ui_config()

        for cmd_type, config in raw_config.items():
            fields = []
            visible_keys = config.get('visible', [])

            # Special case for output producers
            produces_output_cmd = config.get('produces_output', False)

            for key in visible_keys:
                field_def = cls._infer_field_def(key, config)

                # Override label from config if present
                label_key = f"label_{key}"
                if label_key in config:
                    field_def.label = config[label_key]

                # Check for explicit enum configuration
                # Format in JSON might be: "field_name_enum": "dm_ai_module.SomeEnum"
                enum_key = f"{key}_enum"
                if enum_key in config:
                    field_def.field_type = FieldType.ENUM
                    field_def.enum_source = config[enum_key]

                # Special handling for output link
                if key == 'output_link' and produces_output_cmd:
                    field_def.produces_output = True

                fields.append(field_def)

            # Ensure 'optional' and 'up_to' are always present for uniformity
            existing_keys = {f.key for f in fields}

            if 'optional' not in existing_keys:
                field_def = cls._infer_field_def('optional', config)
                if 'label_optional' in config:
                    field_def.label = config['label_optional']
                fields.append(field_def)

            if 'up_to' not in existing_keys:
                # Only add 'up_to' if 'amount' is present, as it implies a quantity limit
                if 'amount' in existing_keys:
                    field_def = cls._infer_field_def('up_to', config)
                    if 'label_up_to' in config:
                        field_def.label = config['label_up_to']
                    fields.append(field_def)

            schema = CommandSchema(type_name=cmd_type, fields=fields, allowed_filter_fields=config.get('allowed_filter_fields', None))
            register_schema(schema)

    @classmethod
    def _infer_field_def(cls, key: str, config: Dict) -> FieldSchema:
        """Infers the FieldSchema from the key name using default mappings."""
        mapping = cls._DEFAULT_MAPPING.get(key, {})

        f_type = mapping.get('type', FieldType.STRING) # Default to String
        label = mapping.get('label', key.replace('_', ' ').title())
        hint = mapping.get('hint')
        min_val = mapping.get('min')

        return FieldSchema(
            key=key,
            label=label,
            field_type=f_type,
            widget_hint=hint,
            min_value=min_val
        )
