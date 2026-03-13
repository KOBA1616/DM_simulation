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
    # 再発防止: プレビュー再構築時に mode/timing/scope/filter が欠落すると
    # 置換文面（〜る時）やスコープ文面が反映されなくなるため、EffectModel に明示定義する。
    mode: Optional[str] = None                  # TRIGGERED / REPLACEMENT
    timing_mode: Optional[str] = None           # PRE / POST
    trigger_scope: str = "NONE"                # NONE / PLAYER_SELF / PLAYER_OPPONENT / ALL_PLAYERS
    trigger_filter: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[ConditionModel] = None # トリガー条件
    commands: List[CommandModel] = Field(default_factory=list) # 実行されるコマンド列

class ModifierModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    type: str = "NONE" # e.g. COST_MODIFIER
    condition: Optional[ConditionModel] = None
    filter: Optional[FilterModel] = None
    value: int = 0
    # Preferred field for keyword/restriction sub-types (kept for backward compatibility with str_val)
    mutation_kind: Optional[str] = None
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

    # Keywords: migrate from Dict to structured KeywordsModel for clarity.
    class KeywordsModel(BaseModel):
        # Special condition entries
        friend_burst_condition: Dict[str, Any] = Field(default_factory=dict)
        revolution_change_condition: Dict[str, Any] = Field(default_factory=dict)
        mekraid_condition: Dict[str, Any] = Field(default_factory=dict)
        # Generic keyword flags (e.g., 's_trigger': True)
        flags: Dict[str, Any] = Field(default_factory=dict)
        # Any other legacy keyword entries
        extras: Dict[str, Any] = Field(default_factory=dict)

        class Config:
            extra = "allow"

        @model_validator(mode='before')
        @classmethod
        def ingest_legacy(cls, v):
            # Accept either dict of keywords or existing KeywordsModel
            if isinstance(v, cls):
                return v
            if isinstance(v, dict):
                # Known condition keys map directly
                kw = {
                    'friend_burst_condition': v.get('friend_burst_condition', {}),
                    'revolution_change_condition': v.get('revolution_change_condition', {}),
                    'mekraid_condition': v.get('mekraid_condition', {}),
                    'flags': {},
                    'extras': {},
                }
                for k, val in v.items():
                    if k in kw:
                        continue
                    # If value is dict and appears condition-like, leave in extras
                    if isinstance(val, dict):
                        kw['extras'][k] = val
                    else:
                        kw['flags'][k] = val
                return kw
            return {'flags': {}, 'extras': {}}

        @model_serializer
        def to_dict(self) -> Dict[str, Any]:
            # Serialize back to plain dict expected by legacy consumers
            out: Dict[str, Any] = {}
            out.update(self.flags or {})
            out.update(self.extras or {})
            if self.friend_burst_condition:
                out['friend_burst_condition'] = self.friend_burst_condition
            if self.revolution_change_condition:
                out['revolution_change_condition'] = self.revolution_change_condition
            if self.mekraid_condition:
                out['mekraid_condition'] = self.mekraid_condition
            return out

    keywords: KeywordsModel = Field(default_factory=KeywordsModel)

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
