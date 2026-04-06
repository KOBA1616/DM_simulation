from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    zones: Optional[List[str]] = Field(default=None)
    civilizations: Optional[List[str]] = Field(default=None)
    civ_match_mode: Optional[str] = Field(default="OR")
    types: Optional[List[str]] = Field(default=None)
    races: Optional[List[str]] = Field(default=None)
    race_match_mode: Optional[str] = Field(default="OR")
    owner: Optional[str] = Field(default=None)
    count: Optional[int] = Field(default=None)
    min_cost: Optional[Union[int, Dict[str, Any]]] = Field(default=None)
    max_cost: Optional[Union[int, Dict[str, Any]]] = Field(default=None)
    exact_cost: Optional[int] = Field(default=None)
    min_power: Optional[Union[int, Dict[str, Any]]] = Field(default=None)
    max_power: Optional[Union[int, Dict[str, Any]]] = Field(default=None)
    exact_power: Optional[int] = Field(default=None)
    min_count: Optional[int] = Field(default=None)
    max_count: Optional[int] = Field(default=None)
    exact_count: Optional[int] = Field(default=None)
    keywords: Optional[List[str]] = Field(default=None)
    extras: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"


def filterspec_from_legacy(d: Optional[Dict[str, Any]]) -> Optional[FilterSpec]:
    """Convert legacy dict to FilterSpec.

    Keeps safe defaults and ignores unknown keys.
    """
    if d is None:
        return None
    if isinstance(d, FilterSpec):
        return d
    if not isinstance(d, dict):
        raise TypeError("expected dict or FilterSpec for conversion")
    # Only pick known fields to avoid accidental data leakage
    keys = {
        'zones', 'civilizations', 'civ_match_mode', 'types', 'races', 'race_match_mode', 'owner', 'count',
        'min_cost', 'max_cost', 'exact_cost',
        'min_power', 'max_power', 'exact_power',
        'min_count', 'max_count', 'exact_count',
        'keywords', 'extras'
    }
    payload = {k: v for k, v in d.items() if k in keys}
    # Ensure extras is a dict when present
    if 'extras' in payload and payload['extras'] is None:
        payload['extras'] = {}
    return FilterSpec(**payload)


def filterspec_to_legacy(f: Optional[FilterSpec]) -> Optional[Dict[str, Any]]:
    """Serialize FilterSpec back to a plain dict compatible with legacy storage.

    This produces a shallow dict of present fields.
    """
    if f is None:
        return None
    if isinstance(f, dict):
        return f
    data: Dict[str, Any] = {}
    for name in (
        'zones', 'civilizations', 'civ_match_mode', 'types', 'races', 'race_match_mode', 'owner', 'count',
        'min_cost', 'max_cost', 'exact_cost',
        'min_power', 'max_power', 'exact_power',
        'min_count', 'max_count', 'exact_count',
        'keywords', 'extras'
    ):
        val = getattr(f, name, None)
        if val is not None:
            data[name] = val
    return data
