import json
from types import SimpleNamespace

class JsonLoader:
    @staticmethod
    def load_cards(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out = {}
        for c in data:
            ns = SimpleNamespace()
            ns.name = c.get('name')
            ns.cost = c.get('cost')
            ns.power = c.get('power')
            civs = c.get('civilizations', [])
            ns.civilizations = [Civilization[v] if isinstance(v, str) and v in Civilization.__members__ else v for v in civs]
            t = c.get('type')
            ns.type = CardType[t] if isinstance(t, str) and t in CardType.__members__ else t
            # expose keywords as attribute-accessible object
            ns.keywords = SimpleNamespace(**c.get('keywords', {}))
            ns.effects = c.get('effects', [])
            out[c.get('id')] = ns
        return out

from enum import Enum

class Civilization(Enum):
    FIRE = 'FIRE'

class CardType(Enum):
    CREATURE = 'CREATURE'
