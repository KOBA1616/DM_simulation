# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr, model_serializer
from typing import List, Optional, Union, Dict, Any, Literal
import uuid

def generate_uid():
    return str(uuid.uuid4())

# --- Primitive Models ---

class ConditionModel(BaseModel):
    type: str = "NONE"
    value: Optional[int] = None
    str_val: Optional[str] = None
    target: Optional[str] = None  # e.g., for target_player in condition
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"

class FilterModel(BaseModel):
    zones: List[str] = Field(default_factory=list)
    civilizations: List[str] = Field(default_factory=list)
    races: List[str] = Field(default_factory=list)
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    min_power: Optional[int] = None
    max_power: Optional[int] = None
    owner: Optional[str] = None
    flags: List[str] = Field(default_factory=list) # e.g. is_tapped

    class Config:
        extra = "allow"

# --- Command Models ---

class CommandModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    type: str  # DRAW_CARD, BREAK_SHIELD etc.
    params: Dict[str, Any] = Field(default_factory=dict) # 汎用パラメータ格納

    # 制御構造 (Composite Pattern)
    if_true: List['CommandModel'] = Field(default_factory=list)
    if_false: List['CommandModel'] = Field(default_factory=list)
    options: List[List['CommandModel']] = Field(default_factory=list) # 選択肢分岐

    # 変数リンク
    input_var: Optional[str] = None
    output_var: Optional[str] = None

    class Config:
        extra = "allow"
        populate_by_name = True

    @model_validator(mode='before')
    @classmethod
    def ingest_legacy_structure(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Map Keys
            # input_link/input_value_key -> input_var
            if 'input_var' not in data:
                if 'input_link' in data:
                    data['input_var'] = data.pop('input_link')
                elif 'input_value_key' in data:
                    data['input_var'] = data.pop('input_value_key')

            if 'output_var' not in data:
                if 'output_link' in data:
                    data['output_var'] = data.pop('output_link')
                elif 'output_value_key' in data:
                    data['output_var'] = data.pop('output_value_key')

            # 2. Move unknown fields to params
            known_fields = {
                'uid', 'type', 'params',
                'if_true', 'if_false', 'options',
                'input_var', 'output_var'
            }

            # If params already exists, use it, else create new
            params = data.get('params', {})
            if not isinstance(params, dict): params = {}

            new_data = {}
            # Copy known fields
            for k in known_fields:
                if k in data:
                    new_data[k] = data[k]

            # Move everything else to params
            for k, v in data.items():
                if k not in known_fields and k != 'params':
                    params[k] = v

            new_data['params'] = params
            return new_data
        return data

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        # 1. Base dump
        # We manually build the dict to control flattening
        result = {
            'uid': self.uid,
            'type': self.type,
        }

        # 2. Flatten params
        if self.params:
            result.update(self.params)

        # 3. Handle recursive fields
        if self.if_true:
            result['if_true'] = [m.model_dump() for m in self.if_true]
        if self.if_false:
            result['if_false'] = [m.model_dump() for m in self.if_false]
        if self.options:
            result['options'] = [[m.model_dump() for m in opt] for opt in self.options]

        # 4. Map Variables to JSON keys
        if self.input_var:
            result['input_value_key'] = self.input_var
            # result['input_link'] = self.input_var # Optional: Keep legacy for python side if needed? No, standardizing.

        if self.output_var:
            result['output_value_key'] = self.output_var

        return result

class EffectModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    trigger: str = "NONE" # ON_PLAY, ON_ATTACK etc.
    condition: Optional[ConditionModel] = None # トリガー条件
    commands: List[CommandModel] = Field(default_factory=list) # 実行されるコマンド列

class ModifierModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    type: str = "NONE" # e.g. COST_MODIFIER
    condition: Optional[ConditionModel] = None
    filter: Optional[FilterModel] = None
    value: int = 0
    str_val: Optional[str] = None
    scope: str = "ALL"

class ReactionModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    type: str = "NONE" # e.g. NINJA_STRIKE
    cost: Optional[int] = None
    zone: Optional[str] = None
    condition: Optional[ConditionModel] = None # e.g. trigger_event inside

# --- Card Model ---

class CardModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    id: int = 0
    name: str = "New Card"
    type: str = "CREATURE"
    civilizations: List[str] = Field(default_factory=lambda: ["FIRE"])
    races: List[str] = Field(default_factory=list)
    cost: int = 1
    power: int = 1000

    keywords: Dict[str, Any] = Field(default_factory=dict)

    effects: List[EffectModel] = Field(default_factory=list)
    static_abilities: List[ModifierModel] = Field(default_factory=list)
    reaction_abilities: List[ReactionModel] = Field(default_factory=list)

    spell_side: Optional['CardModel'] = None

    # Helper fields for editor logic (legacy cleanup)
    # Using PrivateAttr so it's not part of the standard schema but accessible
    _editor_warnings: List[str] = PrivateAttr(default_factory=list)

    @field_validator('civilizations', mode='before')
    def parse_civs(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    class Config:
        extra = "allow"
