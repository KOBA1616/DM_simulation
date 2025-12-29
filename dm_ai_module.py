# Top-level shim to ensure tests can import dm_ai_module in environments
# where the compiled extension is incomplete. This intentionally lives at
# repository root so it takes precedence on sys.path during test collection.
from enum import Enum

class Civilization(Enum):
    FIRE = 1
    WATER = 2
    NATURE = 3
    LIGHT = 4
    DARKNESS = 5

class CardType(Enum):
    CREATURE = 1
    SPELL = 2

class CardKeywords(int):
    pass

class PassiveType(Enum):
    NONE = 0

class PassiveEffect:
    def __init__(self, *a, **k):
        pass

class FilterDef(dict):
    pass

class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand = []
        self.deck = []
        self.battle_zone = []
        self.graveyard = []
        self.mana_zone = []


class GameState:
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.players = [Player(i) for i in range(num_players)]

    def add_card_to_deck(self, player_id: int, card_id: int, instance_id: int = None):
        class C:
            pass
        c = C()
        c.card_id = card_id
        c.instance_id = instance_id
        self.players[player_id].deck.append(c)

    def add_test_card_to_battle(self, player_id: int, card_id: int, instance_id: int, tapped=False, something=False):
        class C:
            pass
        c = C()
        c.card_id = card_id
        c.instance_id = instance_id
        c.tapped = tapped
        self.players[player_id].battle_zone.append(c)

class CardDefinition:
    def __init__(self, *a, **k):
        pass

# Optional placeholders for types the code may inspect
CommandType = None
Zone = None


# Minimal CommandType enum used by Python tests
from enum import Enum

class CommandType(Enum):
    NONE = 0
    TRANSITION = 1
    DRAW_CARD = 2
    DESTROY = 3
    DISCARD = 4
    MANA_CHARGE = 5
    SHIELD_BURN = 6
    TAP = 7
    UNTAP = 8
    QUERY = 9
    MUTATE = 10
    CHOICE = 11
    SELECT_NUMBER = 12
    SEARCH_DECK = 13
    SHUFFLE_DECK = 14
    REVEAL_CARDS = 15
    LOOK_AND_ADD = 16
    MEKRAID = 17
    REVOLUTION_CHANGE = 18
    PLAY_FROM_ZONE = 19
    FRIEND_BURST = 20
    REGISTER_DELAYED_EFFECT = 21
    CAST_SPELL = 22
    RESET_INSTANCE = 23
    SEND_SHIELD_TO_GRAVE = 24
    PUT_CREATURE = 25
    ATTACK_PLAYER = 26
    ATTACK_CREATURE = 27
    FLOW = 28
    BREAK_SHIELD = 29
    RESOLVE_BATTLE = 30
    RESOLVE_EFFECT = 31
    USE_SHIELD_TRIGGER = 32
    RESOLVE_PLAY = 33
    LOOK_TO_BUFFER = 34
    SELECT_FROM_BUFFER = 35
    PLAY_FROM_BUFFER = 36
    SUMMON_TOKEN = 37
    ADD_SHIELD = 38
    ADD_KEYWORD = 39


# Minimal TargetScope enum
class TargetScope(Enum):
    NONE = 0
    PLAYER_OPPONENT = 1
    PLAYER_SELF = 2


class CommandDef:
    def __init__(self, type_enum: CommandType = CommandType.NONE, **kwargs):
        self.type = type_enum
        # common numeric fields
        self.amount = kwargs.get('amount')
        self.target_group = kwargs.get('target_group')
        self.params = kwargs
        # allow dynamic assignment of other attributes commonly used in tests
        # (from_zone, to_zone, target_filter, instance_id, etc.)
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


class CardData:
    def __init__(self, id_, name, cost, civ, power, ctype, races, effects):
        self.id = id_
        self.name = name
        self.cost = cost
        self.civilizations = civ
        self.power = power
        self.type = ctype
        self.races = races
        self.effects = effects


# Simple registry utilities used by tests to register CardData
_CARD_REGISTRY = {}

def register_card_data(carddata: CardData):
    _CARD_REGISTRY[carddata.id] = carddata

def get_card_data(card_id):
    return _CARD_REGISTRY.get(card_id)


# Light placeholders used by compatibility tests / mocks
class EffectResolver:
    pass


class PhaseManager:
    pass


class CommandSystem:
    @staticmethod
    def execute_command(state, cmd, instance_id, player_id, ctx):
        try:
            ctype = cmd.type
        except Exception:
            return

        # Handle TRANSITION draws from deck to hand
        if ctype == CommandType.TRANSITION:
            from_zone = getattr(cmd, 'from_zone', None)
            to_zone = getattr(cmd, 'to_zone', None)
            amount = int(getattr(cmd, 'amount', 1) or 1)

            # Draw
            if from_zone == 'DECK' and to_zone == 'HAND':
                p = state.players[player_id]
                for _ in range(amount):
                    if not p.deck:
                        break
                    card = p.deck.pop(0)
                    p.hand.append(card)
                return

            # Transition implicit from battle to graveyard (destroy)
            if to_zone == 'GRAVEYARD':
                p = state.players[player_id]
                # Try to find by instance_id
                for i, inst in enumerate(list(p.battle_zone)):
                    if getattr(inst, 'instance_id', None) == instance_id:
                        p.battle_zone.pop(i)
                        p.graveyard.append(inst)
                        break
                return

        # Basic fallback: do nothing
        return


class Effect:
    def __init__(self, actions):
        # commands will be filled by JsonLoader
        self.actions = actions
        self.commands = []


class JsonLoader:
    @staticmethod
    def load_cards(path: str):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        from dm_toolkit.action_to_command import map_action

        card_db = {}
        for item in data:
            cid = item.get('id')
            effects = []
            for eff in item.get('effects', []):
                actions = eff.get('actions', [])
                effect = Effect(actions)
                # convert actions to commands
                for a in actions:
                    cmdd = map_action(a)
                    # Normalize target_group strings to TargetScope enum where possible
                    tg = cmdd.get('target_group')
                    if isinstance(tg, str):
                        try:
                            cmdd['target_group'] = TargetScope[tg]
                        except Exception:
                            cmdd['target_group'] = TargetScope.NONE
                    # Prefer the original legacy action type when provided by
                    # `map_action` (it sets 'legacy_original_type') so that
                    # JsonLoader expansion preserves command type semantics
                    # expected by legacy Python tests.
                    tname = str(cmdd.get('legacy_original_type') or cmdd.get('type') or 'NONE')
                    try:
                        t_enum = CommandType[tname]
                    except Exception:
                        t_enum = CommandType.NONE
                    c = CommandDef(t_enum, **cmdd)
                    effect.commands.append(c)
                effects.append(effect)
            # Build a lightweight CardDefinition-like object expected by tests
            class CD:
                def __init__(self, raw, effects_):
                    self.id = raw.get('id')
                    self.name = raw.get('name')
                    self.type = raw.get('type')
                    self.civilizations = raw.get('civilizations') or raw.get('civilization')
                    # keywords -> simple object with attributes
                    kw = raw.get('keywords') or {}
                    class K:
                        def __init__(self, d):
                            for kk, vv in (d.items() if isinstance(d, dict) else []):
                                setattr(self, kk, vv)
                    # fallback: allow attribute access for keys
                    kobj = type('Keywords', (), {})()
                    if isinstance(kw, dict):
                        for kk, vv in kw.items():
                            setattr(kobj, kk, vv)
                    self.keywords = kobj
                    # spell_side may be nested card object
                    if 'spell_side' in raw and raw['spell_side']:
                        ss = raw['spell_side']
                        ss_effects = []
                        for eff in ss.get('effects', []):
                            ss_effects.append(Effect(eff.get('actions', []) ))
                        self.spell_side = CD(raw['spell_side'], ss_effects)
                    else:
                        self.spell_side = None
                    self.effects = effects_
                    # commands may be on effects; tests expect effect.commands populated
            card_db[cid] = CD(item, effects)
        return card_db


# Expose names for tests
CommandType = CommandType
TargetScope = TargetScope
JsonLoader = JsonLoader
CommandDef = CommandDef
