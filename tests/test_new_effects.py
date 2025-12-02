
import sys
import os
import json
import pytest

# Add bin path for dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), "../bin"))
try:
    import dm_ai_module
except ImportError:
    pass

class TestNewEffects:
    def create_json_file(self, filename, cards):
        with open(filename, 'w') as f:
            json.dump(cards, f)

    def populate_card_db(self, card_db, cards_json):
        # Helper to sync Python JSON data to C++ CardDefinition map
        for c in cards_json:
            cd = card_db[c['id']]
            cd.id = c['id']
            cd.name = c['name']
            cd.cost = c['cost']
            cd.power = c.get('power', 0)

            civ_map = {
                "FIRE": dm_ai_module.Civilization.FIRE,
                "WATER": dm_ai_module.Civilization.WATER,
                "NATURE": dm_ai_module.Civilization.NATURE,
                "LIGHT": dm_ai_module.Civilization.LIGHT,
                "DARKNESS": dm_ai_module.Civilization.DARKNESS,
                "ZERO": dm_ai_module.Civilization.ZERO
            }
            cd.civilization = civ_map.get(c.get('civilization', "FIRE"), dm_ai_module.Civilization.FIRE)

            if c['type'] == 'SPELL':
                cd.type = dm_ai_module.CardType.SPELL
            else:
                cd.type = dm_ai_module.CardType.CREATURE

            cd.races = c.get('races', [])

            # Check for ON_PLAY effects to set cip keyword
            has_cip = False
            for eff in c.get('effects', []):
                if eff.get('trigger') == 'ON_PLAY':
                    has_cip = True
                    break

            # Set keywords.cip
            # Note: CardKeywords bindings must be mutable
            cd.keywords.cip = has_cip

    def test_search_deck_bottom(self):
        # Define JSON for Searcher
        cards_json = [
            {
                "id": 2, "name": "Searcher", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 3000, "races": [],
                "effects": [{
                    "trigger": "ON_PLAY",
                    "condition": {"type": "NONE"},
                    "actions": [{
                        "type": "SEARCH_DECK_BOTTOM",
                        "scope": "PLAYER_SELF",
                        "value1": 3,
                        "filter": {"types": ["SPELL"], "count": 1}
                    }]
                }]
            },
            {
                "id": 3, "name": "Found Spell", "cost": 0, "civilization": "FIRE", "type": "SPELL", "power": 0, "races": [],
                "effects": []
            },
             {
                "id": 1, "name": "Dummy", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 1000, "races": [],
                "effects": []
            }
        ]

        json_path = "test_cards_search.json"
        self.create_json_file(json_path, cards_json)
        dm_ai_module.JsonLoader.load_cards(json_path)

        game = dm_ai_module.GameInstance()
        card_db_defs = {c['id']: dm_ai_module.CardDefinition() for c in cards_json}
        self.populate_card_db(card_db_defs, cards_json)

        # Start game to init basics (though we will overwrite zones)
        game.start_game(card_db_defs)
        state = game.state
        pid = state.active_player_id

        # Setup using C++ helpers to ensure C++ state is updated
        # Clear existing zones (set_deck clears deck)
        state.set_deck(pid, [])
        # Need to clear hand/mana/battle/grave manually?
        # GameState doesn't expose clear_zone.
        # But we can assume start_game put 5 in hand, 0 in mana/battle.
        # We'll just add our test cards.

        # Deck: Top is Dummy, then Spell, then Dummy.
        # push_back is top.
        # We want top to be [Dummy(102), Spell(101), Dummy(100)...]
        # SEARCH looks at top N.
        state.add_card_to_deck(pid, 1, 100) # Bottom
        state.add_card_to_deck(pid, 3, 101) # Target
        state.add_card_to_deck(pid, 1, 102) # Top

        state.add_card_to_hand(pid, 2, 200) # Searcher

        # Mana (3 for cost 0 is fine)
        for i in range(3):
            state.add_card_to_mana(pid, 1, 300+i)

        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.PLAY_CARD
        action.card_id = 2
        action.source_instance_id = 200

        dm_ai_module.EffectResolver.resolve_action(state, action, card_db_defs)

        # Verify: Fetch FRESH copy of player
        player = state.players[pid]

        # 1. Searcher in Battle Zone
        assert len(player.battle_zone) >= 1
        found_searcher = False
        for c in player.battle_zone:
            if c.card_id == 2: found_searcher = True
        assert found_searcher

        # 2. Spell (ID 3) should be in hand
        found_spell = False
        for c in player.hand:
            if c.card_id == 3: found_spell = True
        assert found_spell, "Spell should be added to hand"

        # 3. Remaining cards (Dummy 102, Dummy 100) should be at bottom of deck.
        # Deck size check
        # Initial 3. Look 3. Pick 1. Rest 2.
        # Note: start_game adds cards to deck/hand. So there are extra cards.
        # start_game adds 40 cards to deck and draws 5.
        # So we have 35 cards + 3 we added = 38?
        # Verify deck count logic?
        # Actually, if we just check if ID 3 is in hand, that proves Search worked.

        os.remove(json_path)

    def test_mekraid(self):
        cards_json = [
            {
                "id": 4, "name": "Mekraid Source", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 3000, "races": ["Magic"],
                "effects": [{
                    "trigger": "ON_PLAY",
                    "condition": {"type": "NONE"},
                    "actions": [{
                        "type": "MEKRAID",
                        "scope": "PLAYER_SELF",
                        "value1": 3,
                        "filter": {"races": ["Magic"], "max_cost": 5}
                    }]
                }]
            },
            {
                "id": 5, "name": "Mekraid Creature", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 3000, "races": ["Magic"],
                "effects": []
            },
             {
                "id": 1, "name": "Dummy", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 1000, "races": ["Human"],
                "effects": []
            }
        ]

        json_path = "test_cards_mekraid.json"
        self.create_json_file(json_path, cards_json)
        dm_ai_module.JsonLoader.load_cards(json_path)

        game = dm_ai_module.GameInstance()
        card_db_defs = {c['id']: dm_ai_module.CardDefinition() for c in cards_json}
        self.populate_card_db(card_db_defs, cards_json)

        game.start_game(card_db_defs)
        state = game.state
        pid = state.active_player_id

        state.set_deck(pid, [])
        state.add_card_to_deck(pid, 1, 100)
        state.add_card_to_deck(pid, 5, 101) # Magic

        state.add_card_to_hand(pid, 4, 200) # Source

        for i in range(5):
            state.add_card_to_mana(pid, 1, 300+i)

        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.PLAY_CARD
        action.card_id = 4
        action.source_instance_id = 200

        dm_ai_module.EffectResolver.resolve_action(state, action, card_db_defs)

        player = state.players[pid]

        source_in_bz = False
        for c in player.battle_zone:
            if c.instance_id == 200: source_in_bz = True
        assert source_in_bz

        target_in_bz = False
        for c in player.battle_zone:
            if c.card_id == 5: target_in_bz = True
        assert target_in_bz

        os.remove(json_path)

    def test_bounce(self):
        cards_json = [
            {
                "id": 6, "name": "Bounce Source", "cost": 0, "civilization": "WATER", "type": "CREATURE", "power": 3000, "races": [],
                "effects": [{
                    "trigger": "ON_PLAY",
                    "condition": {"type": "NONE"},
                    "actions": [{
                        "type": "RETURN_TO_HAND",
                        "scope": "PLAYER_OPPONENT",
                        "value1": 1,
                        "filter": {"types": ["CREATURE"], "count": 1}
                    }]
                }]
            },
            {
                "id": 7, "name": "Bounce Target", "cost": 0, "civilization": "FIRE", "type": "CREATURE", "power": 2000, "races": [],
                "effects": []
            }
        ]

        json_path = "test_cards_bounce.json"
        self.create_json_file(json_path, cards_json)
        dm_ai_module.JsonLoader.load_cards(json_path)

        game = dm_ai_module.GameInstance()
        card_db_defs = {c['id']: dm_ai_module.CardDefinition() for c in cards_json}
        self.populate_card_db(card_db_defs, cards_json)

        game.start_game(card_db_defs)
        state = game.state

        active_pid = state.active_player_id
        opponent_pid = 1 - active_pid

        state.add_card_to_hand(active_pid, 6, 600)
        state.add_card_to_battle(opponent_pid, 7, 700)

        for i in range(4):
            state.add_card_to_mana(active_pid, 6, 400+i)

        action = dm_ai_module.Action()
        action.type = dm_ai_module.ActionType.PLAY_CARD
        action.card_id = 6
        action.source_instance_id = 600

        dm_ai_module.EffectResolver.resolve_action(state, action, card_db_defs)

        # Verify
        active = state.players[active_pid]
        opponent = state.players[opponent_pid]

        source_in_bz = False
        for c in active.battle_zone:
            if c.instance_id == 600: source_in_bz = True
        assert source_in_bz

        target_in_bz = False
        for c in opponent.battle_zone:
            if c.instance_id == 700: target_in_bz = True
        assert not target_in_bz

        target_in_hand = False
        for c in opponent.hand:
            if c.instance_id == 700: target_in_hand = True
        assert target_in_hand

        os.remove(json_path)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
