# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItem
from PyQt6.QtCore import Qt
from typing import Type, TypeVar, Optional, Union, Dict, Any
from pydantic import BaseModel
from dm_toolkit.gui.editor.consts import ROLE_DATA, ROLE_TYPE

T = TypeVar('T', bound=BaseModel)

class ModelAwareItem(QStandardItem):
    """
    A QStandardItem that is aware of Pydantic Models.
    It stores data as dict in Qt's generic storage for compatibility,
    but provides methods to get/set Pydantic models directly.
    """
    def __init__(self, label: str = "", model_obj: Optional[BaseModel] = None, role_type: str = ""):
        super().__init__(label)
        if role_type:
            self.setData(role_type, ROLE_TYPE)
        if model_obj:
            self.set_model_data(model_obj)

    def set_model_data(self, model_obj: BaseModel):
        """Stores the model data. Converts to dict for Qt storage."""
        if not model_obj:
            self.setData({}, ROLE_DATA)
            return

        if hasattr(model_obj, 'model_dump'):
            data = model_obj.model_dump(by_alias=True)
        else:
            data = model_obj.dict(by_alias=True)
        self.setData(data, ROLE_DATA)

    def get_model_data(self, model_cls: Type[T]) -> Optional[T]:
        """Retrieves data as a specific Pydantic model instance."""
        raw_data = self.data(ROLE_DATA)
        if not raw_data:
            return None
        try:
            return model_cls(**raw_data)
        except Exception:
            # If validation fails, fallback to construct (bypass validation) or return None
            # depending on strictness requirements. For now, try construct.
            try:
                return model_cls.construct(**raw_data)
            except Exception:
                return None

def set_item_model(item: QStandardItem, model_obj: BaseModel):
    """Helper to set model data on any QStandardItem."""
    if hasattr(item, 'set_model_data'):
        item.set_model_data(model_obj)
    else:
        if not model_obj:
            item.setData({}, ROLE_DATA)
            return
        if hasattr(model_obj, 'model_dump'):
            data = model_obj.model_dump(by_alias=True)
        else:
            data = model_obj.dict(by_alias=True)
        item.setData(data, ROLE_DATA)

def get_item_model(item: QStandardItem, model_cls: Type[T]) -> Optional[T]:
    """Helper to get model data from any QStandardItem."""
    if hasattr(item, 'get_model_data'):
        return item.get_model_data(model_cls)

    raw_data = item.data(ROLE_DATA)
    if not raw_data:
        return None
    try:
        return model_cls(**raw_data)
    except Exception:
        try:
            return model_cls.construct(**raw_data)
        except Exception:
            return None
