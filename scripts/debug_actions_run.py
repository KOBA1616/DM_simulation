import sys
import os
import json
from types import SimpleNamespace

# Ensure repository root is on sys.path so local modules are importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dm_ai_module
from dm_ai_module import PhaseManager, ActionGenerator

def load_card_db():
    # Try module loader first
    if hasattr(dm_ai_module, 'JsonLoader'):
        try:
            return dm_ai_module.JsonLoader.load_cards(os.path.join('data', 'cards.json'))
        except Exception:
            pass
    # Fallback: parse JSON and construct CardDefinition when possible
    try:
        with open(os.path.join(ROOT, 'data', 'cards.json'), 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception:
        return {}

    card_db = {}
    def make_cd(k, v):
        try:
            if hasattr(dm_ai_module, 'CardDefinition'):
                kws = v.get('keywords', {}) or {}
                cd = dm_ai_module.CardDefinition(int(k), v.get('name', ''), None, v.get('civilizations', []) or [], v.get('races', []) or [], v.get('cost', 0) or 0, v.get('power', 0) or 0, dm_ai_module.CardKeywords(kws), [])
                t = v.get('type')
                if t is not None:
                    try:
                        cd.type = dm_ai_module.CardType[t]
                    except Exception:
                        cd.type = t
                return cd
        except Exception:
            pass
        return SimpleNamespace(type=v.get('type'), cost=v.get('cost', 0))

    if isinstance(raw, list):
        for item in raw:
            cid = item.get('id') or item.get('card_id')
            if cid is None:
                continue
            card_db[int(cid)] = make_cd(cid, item)
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                card_db[int(k)] = make_cd(k, v)
    return card_db

def ensure_players_and_zones(state, config):
    def _ensure(s, pid):
        if hasattr(s, '_ensure_player'):
            try:
                s._ensure_player(pid)
                return
            except Exception:
                pass
        while len(getattr(s, 'players', []) or []) <= pid:
            ns = SimpleNamespace()
            ns.id = len(getattr(s, 'players', []) or [])
            ns.hand = []
            ns.mana_zone = []
            ns.shield = []
            ns.shield_zone = ns.shield
            ns.battle = []
            ns.battle_zone = ns.battle
            ns.deck = []
            ns.graveyard = []
            try:
                s.players.append(ns)
            except Exception:
                s.players = [ns]

    _ensure(state, 0)
    _ensure(state, 1)

    # hand
    for i, cid in enumerate(getattr(config, 'my_hand_cards', []) or []):
        try:
            iid = 100 + i
            player = state.players[0]
            inst = SimpleNamespace(card_id=int(cid), id=iid, instance_id=iid, is_tapped=False)
            try:
                if hasattr(state, 'add_card_to_hand'):
                    state.add_card_to_hand(0, int(cid), iid)
                else:
                    player.hand.append(inst)
            except Exception:
                player.hand.append(inst)
        except Exception:
            pass

    # mana
    for i, cid in enumerate(getattr(config, 'my_mana_zone', []) or []):
        try:
            iid = 200 + i
            player = state.players[0]
            inst = SimpleNamespace(card_id=int(cid), id=iid, instance_id=iid, civilizations=[])
            try:
                if hasattr(state, 'add_card_to_mana'):
                    state.add_card_to_mana(0, int(cid), iid)
                else:
                    player.mana_zone.append(inst)
            except Exception:
                player.mana_zone.append(inst)
        except Exception:
            pass

def main():
    card_db = load_card_db()
    creature_id = -1
    for cid, card in card_db.items():
        try:
            if getattr(card, 'type', None) == dm_ai_module.CardType.CREATURE and int(getattr(card, 'cost', 999)) <= 2:
                creature_id = cid
                break
        except Exception:
            pass

    print('creature_id:', creature_id)
    state = dm_ai_module.GameState(1000)
    if hasattr(dm_ai_module, 'ScenarioConfig'):
        config = dm_ai_module.ScenarioConfig()
    else:
        config = SimpleNamespace()
    config.my_hand_cards = [creature_id]
    config.my_mana_zone = [creature_id] * 5

    if hasattr(PhaseManager, 'setup_scenario'):
        PhaseManager.setup_scenario(state, config, card_db)
    else:
        ensure_players_and_zones(state, config)

    print('hand len', len(state.players[0].hand), 'mana len', len(state.players[0].mana_zone))

    # Ensure card_db values are CardDefinition instances when required by compiled stub
    if hasattr(dm_ai_module, 'CardDefinition'):
        try:
            new_db = {}
            for k, v in list(card_db.items()):
                try:
                    if isinstance(v, dm_ai_module.CardDefinition):
                        new_db[int(k)] = v
                        continue
                except Exception:
                    pass
                try:
                    name = getattr(v, 'name', '')
                    civs = getattr(v, 'civilizations', []) or []
                    races = getattr(v, 'races', []) or []
                    cost = int(getattr(v, 'cost', 0) or 0)
                    power = int(getattr(v, 'power', 0) or 0)
                    kws = getattr(v, 'keywords', {}) or {}
                    cd = dm_ai_module.CardDefinition(int(k), name, None, civs, races, cost, power, dm_ai_module.CardKeywords(kws), [])
                    t = getattr(v, 'type', None)
                    try:
                        if t is not None:
                            try:
                                cd.type = dm_ai_module.CardType[t]
                            except Exception:
                                cd.type = t
                    except Exception:
                        pass
                    new_db[int(k)] = cd
                except Exception:
                    pass
            card_db = new_db
        except Exception:
            pass

    actions = ActionGenerator.generate_legal_actions(state, card_db)
    print('actions count', len(actions))
    try:
        from dm_toolkit.commands_new import generate_legal_commands
    except Exception:
        generate_legal_commands = None
    if generate_legal_commands:
        try:
            cmds = generate_legal_commands(state, card_db) or []
            print('commands count', len(cmds))
        except Exception:
            cmds = []
    else:
        cmds = []
    for i, a in enumerate(actions):
        print(i, getattr(a, 'type', None), getattr(a, 'card_id', None), getattr(a, 'command', None))

if __name__ == '__main__':
    main()
