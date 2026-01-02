from debug_actions_run import main

if __name__ == '__main__':
    main()
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

# Load card DB via dm_ai_module.JsonLoader when available, otherwise fall back
if hasattr(dm_ai_module, 'JsonLoader'):
    try:
        card_db = dm_ai_module.JsonLoader.load_cards(os.path.join('data', 'cards.json'))
    except Exception:
        card_db = {}
else:
    try:
        with open(os.path.join(ROOT, 'data', 'cards.json'), 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            card_db = {}
            for item in raw:
                from debug_actions_run import main

                if __name__ == '__main__':
                    main()
        card_db = new_db
    except Exception:
        pass

actions = ActionGenerator.generate_legal_actions(state, card_db)
print('actions count', len(actions))
try:
    from dm_toolkit.commands import generate_legal_commands
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

# Load card DB via dm_ai_module.JsonLoader when available, otherwise fall back
if hasattr(dm_ai_module, 'JsonLoader'):
    try:
        card_db = dm_ai_module.JsonLoader.load_cards(os.path.join('data', 'cards.json'))
    except Exception:
        card_db = {}
else:
    try:
        with open(os.path.join(ROOT, 'data', 'cards.json'), 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            card_db = {}
            for item in raw:
                cid = item.get('id') or item.get('card_id')
                if cid is None:
                    continue
                ctype = item.get('type')
                try:
                    if ctype == 'CREATURE':
                        ctype_val = dm_ai_module.CardType.CREATURE
                    else:
                        ctype_val = getattr(dm_ai_module.CardType, ctype) if hasattr(dm_ai_module, 'CardType') and ctype is not None else ctype
                except Exception:
                    ctype_val = ctype
                # Prefer constructing a CardDefinition when available to satisfy generate_legal_actions
                try:
                    if hasattr(dm_ai_module, 'CardDefinition'):
                        kws = item.get('keywords', {}) or {}
                        cd = dm_ai_module.CardDefinition(int(cid), item.get('name', ''), None, item.get('civilizations', []) or [], item.get('races', []) or [], item.get('cost', 0) or 0, item.get('power', 0) or 0, dm_ai_module.CardKeywords(kws), [])
                        # Normalize type if present
                        try:
                            if item.get('type') is not None:
                                try:
                                    cd.type = dm_ai_module.CardType[item.get('type')]
                                except Exception:
                                    cd.type = item.get('type')
                        except Exception:
                            pass
                        obj = cd
                    else:
                        obj = SimpleNamespace(type=ctype_val, cost=item.get('cost', 0))
                except Exception:
                    obj = SimpleNamespace(type=ctype_val, cost=item.get('cost', 0))
                card_db[int(cid)] = obj
        elif isinstance(raw, dict):
            card_db = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    ctype = v.get('type')
                    try:
                        if ctype == 'CREATURE':
                            ctype_val = dm_ai_module.CardType.CREATURE
                        else:
                            ctype_val = getattr(dm_ai_module.CardType, ctype) if hasattr(dm_ai_module, 'CardType') and ctype is not None else ctype
                    except Exception:
                        ctype_val = ctype
                    try:
                        if hasattr(dm_ai_module, 'CardDefinition'):
                            kws = v.get('keywords', {}) or {}
                            cd = dm_ai_module.CardDefinition(int(k), v.get('name', ''), None, v.get('civilizations', []) or [], v.get('races', []) or [], v.get('cost', 0) or 0, v.get('power', 0) or 0, dm_ai_module.CardKeywords(kws), [])
                            try:
                                if v.get('type') is not None:
                                    try:
                                        cd.type = dm_ai_module.CardType[v.get('type')]
                                    except Exception:
                                        cd.type = v.get('type')
                            except Exception:
                                pass
                            card_db[int(k)] = cd
                        else:
                            obj = SimpleNamespace(type=ctype_val, cost=v.get('cost', 0))
                            card_db[int(k)] = obj
                    except Exception:
                        card_db[k] = SimpleNamespace(type=ctype_val, cost=v.get('cost', 0))
        else:
            card_db = {}
    except Exception:
        card_db = {}

creature_id = -1
for cid, card in card_db.items():
    try:
        if card.type == dm_ai_module.CardType.CREATURE and int(getattr(card, 'cost', 999)) <= 2:
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

# Use PhaseManager.setup_scenario if available; otherwise perform minimal setup
if hasattr(PhaseManager, 'setup_scenario'):
    PhaseManager.setup_scenario(state, config, card_db)
else:
    # Minimal scenario population so ActionGenerator can operate
    def _ensure_player_local(s, player_id):
        if hasattr(s, '_ensure_player'):
            try:
                s._ensure_player(player_id)
                return
            except Exception:
                pass
        while len(getattr(s, 'players', []) or []) <= player_id:
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
                # If players attribute is not a list, set it
                s.players = [ns]

    _ensure_player_local(state, 0)
    _ensure_player_local(state, 1)

    # Populate hand (append minimal instance objects)
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

    # Populate mana zone
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
print('hand len', len(state.players[0].hand), 'mana len', len(state.players[0].mana_zone))
actions = ActionGenerator.generate_legal_actions(state, card_db)
print('actions count', len(actions))
try:
    from dm_toolkit.commands import generate_legal_commands
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
import dm_ai_module
from dm_ai_module import PhaseManager, ActionGenerator
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
creature_id = -1
for cid, card in card_db.items():
    import sys
    import os

    # Ensure repository root is on sys.path so local modules are importable
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import dm_ai_module
    from dm_ai_module import PhaseManager, ActionGenerator

    card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
    creature_id = -1
    for cid, card in card_db.items():
        import dm_ai_module
        from dm_ai_module import PhaseManager, ActionGenerator
        import json
        from types import SimpleNamespace

        # Load card DB via dm_ai_module.JsonLoader when available, otherwise fall back
        if hasattr(dm_ai_module, 'JsonLoader'):
            try:
                card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
            except Exception:
                card_db = {}
        else:
            # Fallback: parse JSON and construct minimal card objects
            try:
                with open(os.path.join(ROOT, 'data', 'cards.json'), 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    card_db = {}
                    for item in raw:
                        cid = item.get('id') or item.get('card_id')
                        if cid is None:
                            continue
                        # Minimal object with .type and .cost used by debug script
                        ctype = item.get('type')
                        try:
                            if ctype == 'CREATURE':
                                ctype_val = dm_ai_module.CardType.CREATURE
                            else:
                                ctype_val = getattr(dm_ai_module.CardType, ctype) if hasattr(dm_ai_module, 'CardType') and ctype is not None else ctype
                        except Exception:
                            ctype_val = ctype
                        obj = SimpleNamespace(type=ctype_val, cost=item.get('cost', 0))
                        card_db[int(cid)] = obj
                elif isinstance(raw, dict):
                    # If dict, try to coerce values to SimpleNamespace if necessary
                    card_db = {}
                    for k, v in raw.items():
                        if isinstance(v, dict):
                            ctype = v.get('type')
                            try:
                                if ctype == 'CREATURE':
                                    ctype_val = dm_ai_module.CardType.CREATURE
                                else:
                                    ctype_val = getattr(dm_ai_module.CardType, ctype) if hasattr(dm_ai_module, 'CardType') and ctype is not None else ctype
                            except Exception:
                                ctype_val = ctype
                            obj = SimpleNamespace(type=ctype_val, cost=v.get('cost', 0))
                            try:
                                card_db[int(k)] = obj
                            except Exception:
                                card_db[k] = obj
                else:
                    card_db = {}
            except Exception:
                card_db = {}
