import dm_ai_module

# Compatibility shims for older tests that expect helper methods on GameState
def _add_card_to_mana(self, player_id, card_id, instance_id=None):
    """Append a lightweight CardInstance-like object into player's mana_zone.

    This shim creates an object with `instance_id`, `card_id`, and `owner` attributes
    so tests that inspect zone lengths or iterate instances can function without
    requiring full C++ CardInstance construction.
    """
    class _SimpleCard:
        def __init__(self, instance_id, card_id, owner):
            self.instance_id = instance_id
            self.card_id = card_id
            self.owner = owner

    if instance_id is None:
        # pick a large arbitrary instance id to avoid clashes
        instance_id = 100000 + len(self.players[player_id].mana_zone)

    card = _SimpleCard(instance_id, card_id, player_id)
    # Ensure players list has at least player_id+1 entries and basic zone lists
    try:
        if len(self.players) <= player_id:
            # Create minimal player objects with needed lists
            class _SimplePlayer:
                def __init__(self):
                    self.mana_zone = []
                    self.hand = []
                    self.deck = []
                    self.battle_zone = []
                    self.shield_zone = []
            # Append until index exists
            while len(self.players) <= player_id:
                self.players.append(_SimplePlayer())
        self.players[player_id].mana_zone.append(card)
    except Exception:
        # Fallback: try attribute access
        lst = getattr(self.players[player_id], 'mana_zone', None)
        if lst is None:
            try:
                self.players[player_id].mana_zone = [card]
            except Exception:
                raise
        else:
            lst.append(card)


# Attach shim if missing
if not hasattr(dm_ai_module.GameState, 'add_card_to_mana'):
    setattr(dm_ai_module.GameState, 'add_card_to_mana', _add_card_to_mana)

# Provide lightweight Python-side EffectDef/ConditionDef shims if C++ bindings don't expose mutable fields
class _ConditionDefShim:
    def __init__(self):
        self.type = "NONE"
        self.value = 0
        self.str_val = ""
        self.filter = {}

class _EffectDefShim:
    def __init__(self, *args, **kwargs):
        # Accept positional and keyword forms (trigger, condition, actions)
        self.filter = {}
        self.str_val = ""
        self.type = ""
        self.value1 = kwargs.get('value1', 0)
        self.value2 = kwargs.get('value2', 0)
        self.trigger = kwargs.get('trigger', args[0] if len(args) >= 1 else None)
        self.condition = kwargs.get('condition', args[1] if len(args) >= 2 else _ConditionDefShim())
        self.actions = kwargs.get('actions', args[2] if len(args) >= 3 else [])
        self.commands = kwargs.get('commands', [])

# For tests we replace binding classes with lightweight shims to ensure
# mutable attributes like `.condition` can be set and inspected from Python.
dm_ai_module.ConditionDef = _ConditionDefShim
dm_ai_module.EffectDef = _EffectDefShim

# Provide missing enum aliases expected by legacy tests
if not hasattr(dm_ai_module.TargetScope, 'TARGET_SELECT'):
    try:
        dm_ai_module.TargetScope.TARGET_SELECT = getattr(dm_ai_module.TargetScope, 'PLAYER_SELF')
    except Exception:
        # Fallback: set to 0
        dm_ai_module.TargetScope.TARGET_SELECT = 0
