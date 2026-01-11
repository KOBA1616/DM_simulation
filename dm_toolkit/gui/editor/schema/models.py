from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field
from .enums import (
    TriggerType, TargetScope, ModifierType, Civilization, CardType,
    EffectPrimitive, CommandType, CostType, ReductionType, ReactionType
)

# Helper type for template compatibility
# Allows strict Enum or placeholder string (e.g. "__CARD_CIVILIZATIONS__")
TemplateStr = str

class FilterDef(BaseModel):
    owner: Optional[str] = None
    zones: List[str] = Field(default_factory=list)
    types: List[str] = Field(default_factory=list)
    # Allow Enum or raw string for placeholders
    civilizations: List[Union[Civilization, TemplateStr]] = Field(default_factory=list)
    races: List[str] = Field(default_factory=list)
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    min_power: Optional[int] = None
    max_power: Optional[int] = None
    is_tapped: Optional[bool] = None
    is_blocker: Optional[bool] = None
    is_evolution: Optional[bool] = None
    is_card_designation: Optional[bool] = None
    count: Optional[int] = None
    selection_mode: Optional[str] = None
    selection_sort_key: Optional[str] = None
    power_max_ref: Optional[str] = None
    and_conditions: List['FilterDef'] = Field(default_factory=list)

class ConditionDef(BaseModel):
    type: str = "NONE"
    value: int = 0
    str_val: str = ""
    stat_key: str = ""
    op: str = ""
    filter: Optional[FilterDef] = None

class ModifierDef(BaseModel):
    type: ModifierType = ModifierType.NONE
    value: int = 0
    str_val: str = ""
    condition: Optional[ConditionDef] = None
    filter: Optional[FilterDef] = None
    uid: Optional[str] = None # Editor only

class CostDef(BaseModel):
    type: CostType
    amount: int
    filter: Optional[FilterDef] = None
    is_optional: bool = False
    cost_id: str = ""

class CostReductionDef(BaseModel):
    type: ReductionType
    unit_cost: CostDef
    reduction_amount: int
    max_units: int = -1
    min_mana_cost: int = 0
    name: str = ""

class ActionDef(BaseModel):
    type: EffectPrimitive = EffectPrimitive.NONE
    scope: TargetScope = TargetScope.NONE
    filter: Optional[FilterDef] = None
    value1: int = 0
    value2: int = 0
    str_val: str = ""
    value: str = ""
    optional: bool = False
    target_player: str = ""
    source_zone: str = ""
    destination_zone: str = ""
    target_choice: str = ""
    input_value_key: str = ""
    output_value_key: str = ""
    inverse_target: bool = False
    condition: Optional[ConditionDef] = None
    options: List[List['ActionDef']] = Field(default_factory=list)
    cast_spell_side: bool = False
    uid: Optional[str] = None # Editor only

class CommandDef(BaseModel):
    type: CommandType = CommandType.NONE
    instance_id: int = 0
    target_instance: int = 0
    owner_id: int = 0
    target_group: TargetScope = TargetScope.NONE
    target_filter: Optional[FilterDef] = None
    amount: int = 0
    str_param: str = ""
    optional: bool = False
    from_zone: str = ""
    to_zone: str = ""
    mutation_kind: str = ""
    condition: Optional[ConditionDef] = None
    if_true: List['CommandDef'] = Field(default_factory=list)
    if_false: List['CommandDef'] = Field(default_factory=list)
    options: List[List['CommandDef']] = Field(default_factory=list) # Added for CHOICE command support
    input_value_key: str = ""
    output_value_key: str = ""
    uid: Optional[str] = None # Editor only
    legacy_warning: bool = False # Editor only

class EffectDef(BaseModel):
    trigger: TriggerType = TriggerType.NONE
    condition: Optional[ConditionDef] = None
    commands: List[CommandDef] = Field(default_factory=list)
    uid: Optional[str] = None # Editor only

class ReactionCondition(BaseModel):
    trigger_event: str = "NONE"
    civilization_match: bool = False
    mana_count_min: int = 0
    same_civilization_shield: bool = False

class ReactionAbility(BaseModel):
    type: ReactionType = ReactionType.NONE
    cost: int = 0
    zone: str = ""
    condition: Optional[ReactionCondition] = None
    uid: Optional[str] = None # Editor only

class CardData(BaseModel):
    id: int
    name: str
    cost: int
    civilizations: List[Union[Civilization, TemplateStr]] = Field(default_factory=list)
    power: int
    type: CardType = CardType.CREATURE
    races: List[str] = Field(default_factory=list)
    effects: List[EffectDef] = Field(default_factory=list) # Mapped to 'triggers' or 'effects'
    static_abilities: List[ModifierDef] = Field(default_factory=list)
    metamorph_abilities: List[EffectDef] = Field(default_factory=list)
    evolution_condition: Optional[FilterDef] = None
    revolution_change_condition: Optional[FilterDef] = None
    keywords: Dict[str, bool] = Field(default_factory=dict)
    reaction_abilities: List[ReactionAbility] = Field(default_factory=list)
    cost_reductions: List[CostReductionDef] = Field(default_factory=list)
    spell_side: Optional['CardData'] = None
    is_key_card: bool = False
    ai_importance_score: int = 0
    uid: Optional[str] = None # Editor only

# Update forward refs
FilterDef.update_forward_refs()
ActionDef.update_forward_refs()
CommandDef.update_forward_refs()
CardData.update_forward_refs()
