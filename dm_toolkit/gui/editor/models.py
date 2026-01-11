from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid

@dataclass
class BaseDataNode:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON dictionary."""
        return {"uid": self.uid}

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        """Create instance from JSON dictionary."""
        instance = cls()
        if 'uid' in data:
            instance.uid = data['uid']
        return instance

    def validate(self) -> List[str]:
        """Validate the data structure."""
        return []

@dataclass
class FilterNode(BaseDataNode):
    zones: List[str] = field(default_factory=list)
    civilizations: List[str] = field(default_factory=list)
    races: List[str] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    min_power: Optional[int] = None
    max_power: Optional[int] = None
    is_tapped: Optional[bool] = None
    is_blocker: Optional[bool] = None
    is_evolution: Optional[bool] = None
    count: int = 0
    selection_mode: Optional[str] = None
    selection_sort_key: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        data = super().to_json()
        if self.zones: data['zones'] = self.zones
        if self.civilizations: data['civilizations'] = self.civilizations
        if self.races: data['races'] = self.races
        if self.types: data['types'] = self.types
        if self.min_cost is not None: data['min_cost'] = self.min_cost
        if self.max_cost is not None: data['max_cost'] = self.max_cost
        if self.min_power is not None: data['min_power'] = self.min_power
        if self.max_power is not None: data['max_power'] = self.max_power
        if self.is_tapped is not None: data['is_tapped'] = self.is_tapped
        if self.is_blocker is not None: data['is_blocker'] = self.is_blocker
        if self.is_evolution is not None: data['is_evolution'] = self.is_evolution
        if self.count > 0: data['count'] = self.count
        if self.selection_mode: data['selection_mode'] = self.selection_mode
        if self.selection_sort_key: data['selection_sort_key'] = self.selection_sort_key
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        instance = super().from_json(data)
        instance.zones = data.get('zones', [])
        instance.civilizations = data.get('civilizations', [])
        instance.races = data.get('races', [])
        instance.types = data.get('types', [])
        instance.min_cost = data.get('min_cost')
        instance.max_cost = data.get('max_cost')
        instance.min_power = data.get('min_power')
        instance.max_power = data.get('max_power')
        instance.is_tapped = data.get('is_tapped')
        instance.is_blocker = data.get('is_blocker')
        instance.is_evolution = data.get('is_evolution')
        instance.count = data.get('count', 0)
        instance.selection_mode = data.get('selection_mode')
        instance.selection_sort_key = data.get('selection_sort_key')
        return instance

@dataclass
class CommandNode(BaseDataNode):
    type: str = "NONE"
    target_group: str = "PLAYER_SELF"
    target_filter: Optional[FilterNode] = None
    amount: int = 0
    val2: int = 0
    str_param: str = ""
    from_zone: str = "NONE"
    to_zone: str = "NONE"
    flags: List[str] = field(default_factory=list)
    optional: bool = False
    options: List[List['CommandNode']] = field(default_factory=list)
    if_true: List['CommandNode'] = field(default_factory=list)
    if_false: List['CommandNode'] = field(default_factory=list)
    # Variable linking
    input_value_key: str = ""
    output_value_key: str = ""

    # Specifics
    mutation_kind: str = ""

    def to_json(self) -> Dict[str, Any]:
        data = super().to_json()
        data.update({
            "type": self.type,
            "target_group": self.target_group,
            "amount": self.amount,
            "val2": self.val2,
            "str_param": self.str_param,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "optional": self.optional
        })
        if self.target_filter:
            data['target_filter'] = self.target_filter.to_json()
        if self.flags:
            data['flags'] = self.flags
        if self.mutation_kind:
            data['mutation_kind'] = self.mutation_kind
        if self.input_value_key:
            data['input_value_key'] = self.input_value_key
        if self.output_value_key:
            data['output_value_key'] = self.output_value_key

        if self.options:
            data['options'] = [[cmd.to_json() for cmd in opt_group] for opt_group in self.options]
        if self.if_true:
            data['if_true'] = [cmd.to_json() for cmd in self.if_true]
        if self.if_false:
            data['if_false'] = [cmd.to_json() for cmd in self.if_false]

        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        instance = super().from_json(data)
        instance.type = data.get('type', "NONE")
        instance.target_group = data.get('target_group', "PLAYER_SELF")
        if data.get('target_filter'):
            instance.target_filter = FilterNode.from_json(data['target_filter'])
        instance.amount = data.get('amount', 0)
        instance.val2 = data.get('val2', 0)
        instance.str_param = data.get('str_param', "")
        instance.from_zone = data.get('from_zone', "NONE")
        instance.to_zone = data.get('to_zone', "NONE")
        instance.flags = data.get('flags', [])
        instance.optional = data.get('optional', False)
        instance.mutation_kind = data.get('mutation_kind', "")
        instance.input_value_key = data.get('input_value_key', "")
        instance.output_value_key = data.get('output_value_key', "")

        if 'options' in data:
            instance.options = []
            for opt_group in data['options']:
                instance.options.append([cls.from_json(cmd) for cmd in opt_group])

        if 'if_true' in data:
            instance.if_true = [cls.from_json(cmd) for cmd in data['if_true']]
        if 'if_false' in data:
            instance.if_false = [cls.from_json(cmd) for cmd in data['if_false']]

        return instance

@dataclass
class EffectNode(BaseDataNode):
    trigger: str = "NONE"
    commands: List[CommandNode] = field(default_factory=list)
    condition: Optional[FilterNode] = None  # Simplified condition handling

    def to_json(self) -> Dict[str, Any]:
        data = super().to_json()
        data['trigger'] = self.trigger
        data['commands'] = [cmd.to_json() for cmd in self.commands]
        # Condition handling needs more detail in real implementation, placeholder for now
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        instance = super().from_json(data)
        instance.trigger = data.get('trigger', "NONE")
        instance.commands = [CommandNode.from_json(cmd) for cmd in data.get('commands', [])]
        return instance

@dataclass
class CardNode(BaseDataNode):
    id: int = 0
    name: str = ""
    civilizations: List[str] = field(default_factory=list)
    type: str = "CREATURE"
    cost: int = 1
    power: int = 1000
    races: List[str] = field(default_factory=list)
    effects: List[EffectNode] = field(default_factory=list)
    keywords: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        data = super().to_json()
        data.update({
            "id": self.id,
            "name": self.name,
            "civilizations": self.civilizations,
            "type": self.type,
            "cost": self.cost,
            "power": self.power,
            "races": self.races,
            "keywords": self.keywords
        })
        # Determine whether to use 'triggers' or 'effects' key based on context logic,
        # but for pure data model, we default to 'effects' or expose a flag.
        # Here we stick to 'effects' for simplicity.
        data['effects'] = [eff.to_json() for eff in self.effects]
        return data

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        instance = super().from_json(data)
        instance.id = data.get('id', 0)
        instance.name = data.get('name', "")
        instance.civilizations = data.get('civilizations', [])
        instance.type = data.get('type', "CREATURE")
        instance.cost = data.get('cost', 1)
        instance.power = data.get('power', 1000)
        instance.races = data.get('races', [])
        instance.keywords = data.get('keywords', {})

        # Load from either effects or triggers
        raw_effects = data.get('effects', []) or data.get('triggers', [])
        instance.effects = [EffectNode.from_json(eff) for eff in raw_effects]
        return instance
