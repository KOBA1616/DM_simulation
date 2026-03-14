from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryParams(BaseModel):
    query_type: str
    min: Optional[int] = Field(default=1)
    max: Optional[int] = Field(default=1)
    filter: Optional[Dict[str, Any]] = None


class TransitionParams(BaseModel):
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    amount: Optional[int] = Field(default=1)
    preserve_order: Optional[bool] = Field(default=False)


class ModifierParams(BaseModel):
    target: Optional[str] = None
    delta: Optional[int] = None
    temporary: Optional[bool] = Field(default=False)
