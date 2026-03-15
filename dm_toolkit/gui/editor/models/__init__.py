# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr, model_serializer
from typing import List, Optional, Union, Dict, Any, Literal
from typing import TYPE_CHECKING
import uuid
from dm_toolkit import consts

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
    # Legacy: `flags` was a loose list of flag names. We keep it for ingest only
    # and map to explicit fields below. Serialization will not emit `flags`.
    flags: List[str] = Field(default_factory=list) # e.g. is_tapped

    # Explicit boolean fields (preferred). Use Optional so absence is distinguishable.
    is_tapped: Optional[bool] = None
    is_blocker: Optional[bool] = None
    is_evolution: Optional[bool] = None
    is_card_designation: Optional[bool] = None
    is_trigger_source: Optional[bool] = None

    class Config:
        extra = "allow"

    @model_validator(mode='before')
    @classmethod
    def ingest_legacy_flags(cls, v):
        # Map legacy `flags` list into explicit boolean fields
        if isinstance(v, dict):
            flags = v.get('flags') or []
            if flags and isinstance(flags, list):
                # Normalized set of names
                flags_set = set(flags)
                if 'is_tapped' in flags_set or 'tapped' in flags_set:
                    v.setdefault('is_tapped', True)
                if 'is_blocker' in flags_set or 'blocker' in flags_set:
                    v.setdefault('is_blocker', True)
                if 'is_evolution' in flags_set or 'evolution' in flags_set:
                    v.setdefault('is_evolution', True)
                if 'is_card_designation' in flags_set or 'card_designation' in flags_set:
                    v.setdefault('is_card_designation', True)
                if 'is_trigger_source' in flags_set or 'trigger_source' in flags_set:
                    v.setdefault('is_trigger_source', True)

            # Remove legacy flags from the dict to avoid serialization
            if 'flags' in v:
                try:
                    del v['flags']
                except Exception:
                    pass
        return v

    @model_serializer
    def serialize_filter(self) -> Dict[str, Any]:
        # Serialize filter without legacy `flags` list. Include explicit fields only when set.
        out: Dict[str, Any] = {}
        if self.zones: out['zones'] = self.zones
        if self.civilizations: out['civilizations'] = self.civilizations
        if self.races: out['races'] = self.races
        if self.min_cost is not None: out['min_cost'] = self.min_cost
        if self.max_cost is not None: out['max_cost'] = self.max_cost
        if self.min_power is not None: out['min_power'] = self.min_power
        if self.max_power is not None: out['max_power'] = self.max_power
        if self.owner is not None: out['owner'] = self.owner

        # Explicit flags
        if self.is_tapped is not None: out['is_tapped'] = self.is_tapped
        if self.is_blocker is not None: out['is_blocker'] = self.is_blocker
        if self.is_evolution is not None: out['is_evolution'] = self.is_evolution
        if self.is_card_designation is not None: out['is_card_designation'] = self.is_card_designation
        if self.is_trigger_source is not None: out['is_trigger_source'] = self.is_trigger_source

        return out


# --- FilterSpec (single canonical filter schema) ---
class FilterSpec(BaseModel):
    zones: List[str] = Field(default_factory=list)
    civilizations: List[str] = Field(default_factory=list)
    races: List[str] = Field(default_factory=list)
    min_cost: Optional[int] = None
    max_cost: Optional[int] = None
    min_power: Optional[int] = None
    max_power: Optional[int] = None
    owner: Optional[str] = None
    is_tapped: Optional[bool] = None
    is_blocker: Optional[bool] = None
    is_evolution: Optional[bool] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


def dict_to_filterspec(d: Dict[str, Any]) -> FilterSpec:
    """Convert legacy filter dict to canonical FilterSpec.

    This function is intentionally conservative: it maps known keys and
    preserves unknown keys under `extras`.
    """
    if not d:
        return FilterSpec()
    extras = {}
    known = {}
    for k, v in d.items():
        if k in ("zones", "civilizations", "races", "owner", "types"):
            known[k] = v
        elif k in ("min_cost", "max_cost", "min_power", "max_power"):
            known[k] = v
        elif k in ("is_tapped", "is_blocker", "is_evolution"):
            # 再発防止: 0/1 以外の数値を bool に丸めると不正値(例: 2)を見逃す。
            # 0/1 のみ互換変換し、それ以外は validator 側で検出できるよう保持する。
            if isinstance(v, (int, float)) and v in (0, 1):
                known[k] = bool(v)
            else:
                known[k] = v
        else:
            extras[k] = v

    return FilterSpec(**known, extras=extras)


def filterspec_to_dict(fs: FilterSpec) -> Dict[str, Any]:
    """Convert `FilterSpec` back to a serializable dict for legacy consumers."""
    out: Dict[str, Any] = {}
    if fs.zones: out['zones'] = fs.zones
    if fs.civilizations: out['civilizations'] = fs.civilizations
    if fs.races: out['races'] = fs.races
    if fs.min_cost is not None: out['min_cost'] = fs.min_cost
    if fs.max_cost is not None: out['max_cost'] = fs.max_cost
    if fs.min_power is not None: out['min_power'] = fs.min_power
    if fs.max_power is not None: out['max_power'] = fs.max_power
    if fs.owner is not None: out['owner'] = fs.owner
    if fs.is_tapped is not None: out['is_tapped'] = fs.is_tapped
    if fs.is_blocker is not None: out['is_blocker'] = fs.is_blocker
    if fs.is_evolution is not None: out['is_evolution'] = fs.is_evolution
    if fs.extras:
        out.update(fs.extras)
    return out


def describe_filterspec(fs_input: Any) -> str:
    """Generate a concise human-readable description from a FilterSpec or legacy dict.

    Output is a semicolon-separated list of present constraints in stable order.
    """
    if fs_input is None:
        return "(no filter)"

    if isinstance(fs_input, dict):
        try:
            fs = dict_to_filterspec(fs_input)
        except Exception:
            return "(invalid filter)"
    elif isinstance(fs_input, FilterSpec):
        fs = fs_input
    else:
        # Try to coerce via Pydantic model if possible
        try:
            fs = dict_to_filterspec(fs_input)
        except Exception:
            return "(invalid filter)"

    parts: List[str] = []
    if fs.zones:
        parts.append(f"Zones: {', '.join(map(str, fs.zones))}")
    if getattr(fs, 'civilizations', None):
        parts.append(f"Civilizations: {', '.join(map(str, fs.civilizations))}")
    if getattr(fs, 'types', None):
        parts.append(f"Types: {', '.join(map(str, fs.types))}")
    if getattr(fs, 'races', None):
        parts.append(f"Races: {', '.join(map(str, fs.races))}")
    if getattr(fs, 'min_cost', None) is not None or getattr(fs, 'max_cost', None) is not None:
        minc = getattr(fs, 'min_cost', '')
        maxc = getattr(fs, 'max_cost', '')
        parts.append(f"Cost: {minc}-{maxc}")
    if getattr(fs, 'owner', None):
        parts.append(f"Owner: {fs.owner}")
    # Flags
    flags = []
    for f in ('is_tapped', 'is_blocker', 'is_evolution'):
        v = getattr(fs, f, None)
        if v:
            flags.append(f)
    if flags:
        parts.append(f"Flags: {', '.join(flags)}")

    # Extras presence
    extras = getattr(fs, 'extras', None)
    if extras:
        parts.append(f"Extras: {len(extras)} items")

    if not parts:
        return "(no constraints)"
    return '; '.join(parts)

# --- Typed Params Models (E-1) ---
class QueryParams(BaseModel):
    # 再発防止: スキーマは str_param キーを使用するため、model_validator で両方のキーを受け入れる
    query_string: Optional[str] = None  # legacy field name
    str_param: Optional[str] = None     # schema key (Query Mode)
    options: List[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def map_schema_to_legacy_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            # Schema produces str_param; map to query_string for compatibility
            if 'str_param' in v and 'query_string' not in v:
                v['query_string'] = v['str_param']
            # Reverse: legacy query_string → str_param
            elif 'query_string' in v and 'str_param' not in v:
                v['str_param'] = v['query_string']
        return v

class TransitionParams(BaseModel):
    target_state: str
    reason: Optional[str] = None

class ModifierParams(BaseModel):
    amount: int
    scope: Optional[str] = None


class MutateParams(BaseModel):
    # Accept either string or consts.MutationKind; validator will coerce when possible.
    mutation_kind: Optional[consts.MutationKind] = None
    amount: Optional[int] = None
    target: Optional[str] = None
    filter: Optional[FilterSpec] = None
    duration: Optional[int] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def coerce_mutation_kind(cls, v):
        # v is the input dict before model construction; ensure mutation_kind in nested dict coerced
        if isinstance(v, dict) and 'mutation_kind' in v:
            mk = v.get('mutation_kind')
            if isinstance(mk, str):
                try:
                    v['mutation_kind'] = consts.MutationKind(mk)
                except Exception:
                    # leave as-is for unknown/legacy values
                    pass
        return v
    @model_serializer
    def serialize_mutate(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # Ensure enum is serialized as its string value for legacy consumers
        if self.mutation_kind is not None:
            # If Enum-like, prefer its value attribute; otherwise fall back to str
            if hasattr(self.mutation_kind, 'value'):
                out['mutation_kind'] = getattr(self.mutation_kind, 'value')
            else:
                out['mutation_kind'] = str(self.mutation_kind)
        if self.amount is not None:
            out['amount'] = self.amount
        if self.target is not None:
            out['target'] = self.target
        if self.filter is not None:
            # Convert FilterSpec to legacy dict
            out['filter'] = filterspec_to_dict(self.filter)
        if self.duration is not None:
            out['duration'] = self.duration
        if self.extras:
            out.update(self.extras)
        return out


class ApplyModifierParams(BaseModel):
    # 再発防止: スキーマは str_param/amount キーを使用; model_validator で両方向をサポート
    modifier_type: Optional[str] = None   # legacy field name
    value: Optional[int] = None           # legacy field name
    str_param: Optional[str] = None       # schema key (Effect ID)
    amount: Optional[int] = None          # schema key (Value)
    duration: Optional[str] = None        # schema key (Duration, SELECT type → string)
    scope: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def map_schema_to_legacy_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            # Schema → legacy
            if 'str_param' in v and 'modifier_type' not in v:
                v['modifier_type'] = v['str_param']
            if 'amount' in v and 'value' not in v:
                v['value'] = v['amount']
            # Legacy → schema
            if 'modifier_type' in v and 'str_param' not in v:
                v['str_param'] = v['modifier_type']
            if 'value' in v and 'amount' not in v:
                v['amount'] = v['value']
        return v


class PlayFromZoneParams(BaseModel):
    # 再発防止: スキーマは from_zone/to_zone/target_filter を使用; model_validator でマッピング
    source_zone: Optional[str] = None      # mapped from from_zone
    destination_zone: Optional[str] = None # mapped from to_zone
    from_zone: Optional[str] = None        # schema key
    to_zone: Optional[str] = None          # schema key
    amount: int = 1
    up_to: Optional[bool] = None
    filter: Optional[Dict[str, Any]] = None          # holds target_filter content
    target_filter: Optional[Dict[str, Any]] = None   # schema key
    play_flags: Optional[bool] = None      # schema key (Play for Free)
    str_param: Optional[str] = None        # schema key (Hint)
    extras: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def map_schema_to_legacy_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            # Map schema zone keys to canonical fields
            if 'from_zone' in v and 'source_zone' not in v:
                v['source_zone'] = v['from_zone']
            if 'to_zone' in v and 'destination_zone' not in v:
                v['destination_zone'] = v['to_zone']
            # Map schema filter key
            if 'target_filter' in v and 'filter' not in v:
                v['filter'] = v['target_filter']
        return v


class CastSpellParams(BaseModel):
    spell_id: Optional[int] = None
    target: Optional[str] = None
    cost: Optional[int] = None
    use_mana_from: Optional[str] = None
    target_group: Optional[str] = None
    target_filter: Optional[Dict[str, Any]] = None
    optional: Optional[bool] = None
    play_flags: Optional[Any] = None
    extras: Dict[str, Any] = Field(default_factory=dict)

class SearchParams(BaseModel):
    amount: int = 1
    destination_zone: str = "HAND"
    up_to: Optional[bool] = None
    # filter can be either dict or FilterModel; keep flexible
    filter: Optional[Dict[str, Any]] = None

class LookAndAddParams(BaseModel):
    look_count: int = 3
    add_count: int = 1
    rest_zone: str = "DECK_BOTTOM"
    # optional filter for candidates
    filter: Optional[Dict[str, Any]] = None

class AddKeywordParams(BaseModel):
    # 再発防止: スキーマは str_val/explicit_self/duration(str) を使用; model_validator で両方向マッピング
    keyword: Optional[str] = None         # legacy field name
    str_val: Optional[str] = None         # schema key (Keyword)
    explicit_self: Optional[bool] = None  # schema key (This Card)
    target: Optional[str] = None
    duration: Optional[str] = None        # schema key → string (SELECT type)
    amount: Optional[int] = None          # schema hidden field
    # legacy compatibility: allow extra fields
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def map_schema_to_legacy_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            # Schema → legacy
            if 'str_val' in v and 'keyword' not in v:
                v['keyword'] = v['str_val']
            # Legacy → schema
            elif 'keyword' in v and 'str_val' not in v:
                v['str_val'] = v['keyword']
        return v

class MekraidParams(BaseModel):
    # 再発防止: スキーマは amount(Level)/val2(Look Count)/select_count を使用; model_validator でマッピング
    reveal_count: int = 3                  # legacy field (= schema val2)
    evolution_cost: Optional[int] = None   # legacy field (= schema amount = Max Cost Level)
    amount: Optional[int] = None           # schema key (Level / Max Cost)
    val2: Optional[int] = None             # schema key (Look Count)
    select_count: Optional[int] = None     # schema key (Select Count)
    filter: Optional[Dict[str, Any]] = None

    @model_validator(mode='before')
    @classmethod
    def map_schema_to_legacy_keys(cls, v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            # Schema → legacy
            if 'val2' in v and v['val2'] is not None:
                v.setdefault('reveal_count', v['val2'])
            if 'amount' in v and v['amount'] is not None:
                v.setdefault('evolution_cost', v['amount'])
            # Legacy → schema
            if 'reveal_count' in v and 'val2' not in v:
                v['val2'] = v['reveal_count']
            if 'evolution_cost' in v and v['evolution_cost'] is not None and 'amount' not in v:
                v['amount'] = v['evolution_cost']
        return v


class RevealCardsParams(BaseModel):
    value1: int = 1
    scope: Optional[str] = None
    input_value_key: Optional[str] = None
    # legacy fields may include other misc keys; keep extras
    extras: Dict[str, Any] = Field(default_factory=dict)


class CountCardsParams(BaseModel):
    filter: Optional[Dict[str, Any]] = None
    scope: Optional[str] = None
    mode: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class AddShieldParams(BaseModel):
    amount: int = 1
    source_zone: Optional[str] = None


class BoostManaParams(BaseModel):
    amount: int = 1
    # future: civilization breakdown or source info
    civ: Optional[str] = None


class DrawCardParams(BaseModel):
    amount: int = 1
    up_to: Optional[bool] = None
    destination: Optional[str] = None


class DiscardParams(BaseModel):
    amount: int = 1
    up_to: Optional[bool] = None
    reason: Optional[str] = None


class MoveCardParams(BaseModel):
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    amount: int = 1
    up_to: Optional[bool] = None


class SummonTokenParams(BaseModel):
    token_id: Optional[str] = None
    amount: int = 1
    extras: Dict[str, Any] = Field(default_factory=dict)


class DeclareNumberParams(BaseModel):
    # DECLARE_NUMBER: declare a numeric choice within a range
    value1: int
    value2: Optional[int] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class PowerModParams(BaseModel):
    # POWER_MOD: adjust power of matching cards
    amount: int = 0
    target_group: Optional[str] = None
    target_filter: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class PutCreatureParams(BaseModel):
    # PUT_CREATURE: place a creature instance from a source (e.g., hand) to battle zone
    # 再発防止: スキーマで編集可能な target_filter/target_group/amount をここで保持しないと、
    # CommandModel 変換時に値が欠落して「保存されない」「生成テキストに反映されない」不具合になる。
    card_id: Optional[int] = None
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    amount: int = 1
    target_group: Optional[str] = None
    target_filter: Optional[Dict[str, Any]] = None
    tapped: Optional[bool] = None
    summoned_for_free: Optional[bool] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class ReplaceCardMoveParams(BaseModel):
    # REPLACE_CARD_MOVE: replace a card move destination with another destination
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    replacement_to_zone: Optional[str] = None
    amount: int = 1
    filter: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class SendShieldToGraveParams(BaseModel):
    # SEND_SHIELD_TO_GRAVE: move opponent's shield to graveyard
    amount: int = 1
    target_group: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class ShieldBurnParams(BaseModel):
    # SHIELD_BURN: burn (reveal and destroy) shield(s)
    amount: int = 1
    target_group: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class LookToBufferParams(BaseModel):
    # LOOK_TO_BUFFER: peek from a zone into a temporary buffer
    from_zone: Optional[str] = "DECK"
    amount: int = 1
    input_var: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class RegisterDelayedEffectParams(BaseModel):
    # REGISTER_DELAYED_EFFECT: register a delayed effect by ID with duration
    str_param: Optional[str] = None  # Effect ID
    amount: int = 1                  # Duration in turns
    extras: Dict[str, Any] = Field(default_factory=dict)


class RevealToBufferParams(BaseModel):
    # REVEAL_TO_BUFFER: reveal cards from a zone into buffer
    from_zone: Optional[str] = "DECK"
    amount: int = 1
    input_var: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class SelectFromBufferParams(BaseModel):
    # SELECT_FROM_BUFFER: select cards from the temporary buffer
    filter: Optional[Dict[str, Any]] = None
    amount: int = 1
    extras: Dict[str, Any] = Field(default_factory=dict)


class MoveBufferToZoneParams(BaseModel):
    # MOVE_BUFFER_TO_ZONE: move cards from buffer to a zone
    to_zone: Optional[str] = "HAND"
    amount: int = 1
    filter: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class MoveBufferRemainToZoneParams(BaseModel):
    # MOVE_BUFFER_REMAIN_TO_ZONE: move the remainder of buffer to a zone
    to_zone: Optional[str] = "HAND"
    filter: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class LockSpellParams(BaseModel):
    # LOCK_SPELL: restrict spells for a duration (used for rule locks)
    target_group: Optional[str] = None
    filter: Optional[Dict[str, Any]] = None
    duration: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class PlayFromBufferParams(BaseModel):
    # PLAY_FROM_BUFFER: play a card from the temporary buffer to a destination
    buffer_index: Optional[int] = None  # index in buffer, if specific
    to_zone: Optional[str] = "BATTLE_ZONE"
    tapped: Optional[bool] = None
    summoned_for_free: Optional[bool] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


class IgnoreAbilityParams(BaseModel):
    # IGNORE_ABILITY: temporarily ignore a specific ability or ability group
    # 再発防止: duration はスキーマで SELECT 型（文字列例 "THIS_TURN"）のため Optional[str] が正しい
    ability_id: Optional[str] = None
    target_group: Optional[str] = None
    duration: Optional[str] = None  # duration key string (e.g. "THIS_TURN", "THIS_BATTLE")
    reason: Optional[str] = None
    extras: Dict[str, Any] = Field(default_factory=dict)


# --- Command Models ---

class CommandModel(BaseModel):
    uid: str = Field(default_factory=generate_uid)
    type: str  # DRAW_CARD, BREAK_SHIELD etc.
    # Params can be either a generic dict (legacy) or a typed params model.
    # We support typed params for high-frequency commands to improve safety.
    params: Union[Dict[str, Any], 'QueryParams', 'TransitionParams', 'ModifierParams', 'MutateParams', 'ApplyModifierParams', 'PlayFromZoneParams', 'CastSpellParams', 'SearchParams', 'LookAndAddParams', 'AddKeywordParams', 'MekraidParams', 'RevealCardsParams', 'CountCardsParams', 'DrawCardParams', 'DiscardParams', 'MoveCardParams', 'AddShieldParams', 'BoostManaParams', 'PowerModParams', 'SummonTokenParams', 'DeclareNumberParams', 'PutCreatureParams', 'ReplaceCardMoveParams', 'SendShieldToGraveParams', 'RegisterDelayedEffectParams', 'LookToBufferParams', 'ShieldBurnParams', 'RevealToBufferParams', 'SelectFromBufferParams', 'MoveBufferToZoneParams', 'MoveBufferRemainToZoneParams', 'LockSpellParams', 'PlayFromBufferParams', 'IgnoreAbilityParams'] = Field(default_factory=dict) # 汎用パラメータ格納

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
            # 再発防止: params が既に Pydantic モデルインスタンスの場合は dict 化せずそのまま保持する
            # (isinstance チェックで BaseModel を直接保持することで、型安全性を維持する)
            if isinstance(params, BaseModel):
                # Already a typed model: preserve it, skip dict-level merging below
                new_data = {}
                for k in known_fields:
                    if k in data:
                        new_data[k] = data[k]
                new_data['params'] = params
                return new_data
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
            # Normalize legacy 'filter' key for command types that expect 'target_filter'
            # e.g., POWER_MOD uses 'target_filter' in its typed params model.
            cmd_type = data.get('type') or data.get('type')
            if isinstance(params, dict) and 'filter' in params:
                if cmd_type == 'POWER_MOD' and 'target_filter' not in params:
                    params['target_filter'] = params.pop('filter')
            # --- E-1: map params to typed models for known command types ---
            cmd_type = new_data.get('type')
            try:
                if isinstance(new_data.get('params'), dict):
                    if cmd_type == 'QUERY':
                        new_data['params'] = QueryParams.model_validate(new_data['params'])
                    elif cmd_type == 'SEARCH_DECK':
                        new_data['params'] = SearchParams.model_validate(new_data['params'])
                    elif cmd_type == 'LOOK_AND_ADD':
                        new_data['params'] = LookAndAddParams.model_validate(new_data['params'])
                    elif cmd_type == 'ADD_KEYWORD':
                        new_data['params'] = AddKeywordParams.model_validate(new_data['params'])
                    elif cmd_type == 'MEKRAID':
                        new_data['params'] = MekraidParams.model_validate(new_data['params'])
                    elif cmd_type == 'REVEAL_CARDS':
                        new_data['params'] = RevealCardsParams.model_validate(new_data['params'])
                    elif cmd_type == 'COUNT_CARDS':
                        new_data['params'] = CountCardsParams.model_validate(new_data['params'])
                    elif cmd_type == 'DRAW_CARD':
                        new_data['params'] = DrawCardParams.model_validate(new_data['params'])
                    elif cmd_type == 'DISCARD':
                        new_data['params'] = DiscardParams.model_validate(new_data['params'])
                    elif cmd_type == 'MOVE_CARD':
                        new_data['params'] = MoveCardParams.model_validate(new_data['params'])
                    elif cmd_type == 'ADD_SHIELD':
                        new_data['params'] = AddShieldParams.model_validate(new_data['params'])
                    elif cmd_type == 'BOOST_MANA':
                        new_data['params'] = BoostManaParams.model_validate(new_data['params'])
                    elif cmd_type == 'TRANSITION':
                        new_data['params'] = TransitionParams.model_validate(new_data['params'])
                    elif cmd_type == 'MODIFY':
                        new_data['params'] = ModifierParams.model_validate(new_data['params'])
                    elif cmd_type == 'MUTATE':
                        new_data['params'] = MutateParams.model_validate(new_data['params'])
                    elif cmd_type == 'APPLY_MODIFIER':
                        new_data['params'] = ApplyModifierParams.model_validate(new_data['params'])
                    elif cmd_type == 'POWER_MOD':
                        new_data['params'] = PowerModParams.model_validate(new_data['params'])
                    elif cmd_type == 'PLAY_FROM_ZONE':
                        new_data['params'] = PlayFromZoneParams.model_validate(new_data['params'])
                    elif cmd_type == 'SEND_SHIELD_TO_GRAVE':
                        new_data['params'] = SendShieldToGraveParams.model_validate(new_data['params'])
                    elif cmd_type == 'SHIELD_BURN':
                        new_data['params'] = ShieldBurnParams.model_validate(new_data['params'])
                    elif cmd_type == 'REPLACE_CARD_MOVE':
                        new_data['params'] = ReplaceCardMoveParams.model_validate(new_data['params'])
                    elif cmd_type == 'LOOK_TO_BUFFER':
                        new_data['params'] = LookToBufferParams.model_validate(new_data['params'])
                    elif cmd_type == 'PLAY_FROM_BUFFER':
                        new_data['params'] = PlayFromBufferParams.model_validate(new_data['params'])
                    elif cmd_type == 'IGNORE_ABILITY':
                        new_data['params'] = IgnoreAbilityParams.model_validate(new_data['params'])
                    elif cmd_type == 'REVEAL_TO_BUFFER':
                        new_data['params'] = RevealToBufferParams.model_validate(new_data['params'])
                    elif cmd_type == 'SELECT_FROM_BUFFER':
                        new_data['params'] = SelectFromBufferParams.model_validate(new_data['params'])
                    elif cmd_type == 'LOCK_SPELL':
                        new_data['params'] = LockSpellParams.model_validate(new_data['params'])
                    elif cmd_type == 'MOVE_BUFFER_TO_ZONE':
                        new_data['params'] = MoveBufferToZoneParams.model_validate(new_data['params'])
                    elif cmd_type == 'MOVE_BUFFER_REMAIN_TO_ZONE':
                        new_data['params'] = MoveBufferRemainToZoneParams.model_validate(new_data['params'])
                    elif cmd_type == 'PUT_CREATURE':
                        new_data['params'] = PutCreatureParams.model_validate(new_data['params'])
                    elif cmd_type == 'CAST_SPELL':
                        new_data['params'] = CastSpellParams.model_validate(new_data['params'])
                    elif cmd_type == 'SUMMON_TOKEN':
                        new_data['params'] = SummonTokenParams.model_validate(new_data['params'])
                    elif cmd_type == 'DECLARE_NUMBER':
                        new_data['params'] = DeclareNumberParams.model_validate(new_data['params'])
                    elif cmd_type == 'REGISTER_DELAYED_EFFECT':
                        new_data['params'] = RegisterDelayedEffectParams.model_validate(new_data['params'])
            except Exception:
                # If conversion fails, keep legacy dict to avoid breaking ingest
                pass
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
            # If params is a Pydantic model (typed), dump it to dict first
            try:
                # model_dump exists on Pydantic BaseModel instances
                if hasattr(self.params, 'model_dump'):
                    result.update(self.params.model_dump())
                elif isinstance(self.params, dict):
                    result.update(self.params)
                else:
                    # Fallback: try to coerce to dict
                    result.update(dict(self.params))
            except Exception:
                # On any unexpected failure, skip flattening to avoid breaking serialization
                pass

        # Backwards-compat: if typed params used 'target_filter', also expose legacy 'filter' key
        if 'target_filter' in result and 'filter' not in result:
            result['filter'] = result['target_filter']

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
