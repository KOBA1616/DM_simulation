# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import List, Optional, Union, Dict, Any, Literal
import uuid

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
    type: str
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Common fields (consolidated from various subtypes to flat model for simplicity in Editor)
    target_group: Optional[str] = None
    target_filter: Optional[FilterModel] = None

    amount: int = 1
    str_param: Optional[str] = None

    from_zone: Optional[str] = None
    to_zone: Optional[str] = None

    mutation_kind: Optional[str] = None

    if_true: List['CommandModel'] = Field(default_factory=list)
    if_false: List['CommandModel'] = Field(default_factory=list)
    options: List[List['CommandModel']] = Field(default_factory=list)

    input_link: Optional[str] = None
    output_link: Optional[str] = None

    class Config:
        extra = "allow"
        populate_by_name = True

class EffectModel(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trigger: str = "NONE"
    condition: Optional[ConditionModel] = None
    commands: List[CommandModel] = Field(default_factory=list)

class ModifierModel(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "NONE" # e.g. COST_MODIFIER
    condition: Optional[ConditionModel] = None
    filter: Optional[FilterModel] = None
    value: int = 0
    str_val: Optional[str] = None
    scope: str = "ALL"

class ReactionModel(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "NONE" # e.g. NINJA_STRIKE
    cost: Optional[int] = None
    zone: Optional[str] = None
    condition: Optional[ConditionModel] = None # e.g. trigger_event inside

# --- Card Model ---

class CardModel(BaseModel):
    uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
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
