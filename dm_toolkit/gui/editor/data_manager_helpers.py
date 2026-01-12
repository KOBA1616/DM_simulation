# Temporary patch file - methods to add to CardDataManager

def get_card_context_type(self, item):
    """Get the 'type' field of a card or spell_side item."""
    item = self._ensure_item(item)
    if not item:
        return "CREATURE"
    data = self.get_item_data(item)
    return data.get('type', 'CREATURE')

def create_default_trigger_data(self):
    """Create default data for a triggered effect."""
    from dm_toolkit.gui.editor.models import EffectModel
    model = EffectModel(
        trigger="ON_PLAY",
        condition=None,
        commands=[]
    )
    return model.model_dump()

def create_default_static_data(self):
    """Create default data for a static ability."""
    from dm_toolkit.gui.editor.models import ModifierModel
    model = ModifierModel(
        type="COST_MODIFIER",
        value=0,
        scope="ALL"
    )
    return model.model_dump()
