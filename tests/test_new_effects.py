
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
        for c in cards_json:
            cd = card_db[c['id']]
            cd.id = c['id']
            cd.name = c['name']
            cd.cost = c['cost']
            cd.power = c.get('power', 0)
            cd.civilization = dm_ai_module.Civilization.FIRE
            cd.type = dm_ai_module.CardType.CREATURE
            cd.races = c.get('races', [])

    def test_bounce(self):
        cards_json = [
            {
                "id": 6, "name": "Bounce Source", "cost": 0, "civilization": "WATER", "type": "CREATURE", "power": 3000, "races": [],
                "effects": [{
                    "trigger": "ON_PLAY",
                    "condition": {"type": "NONE"},
                    "actions": [{
                        "type": "RETURN_TO_HAND",
                        "scope": "TARGET_SELECT",
                        "value1": 1,
                        "filter": {"zones": ["BATTLE_ZONE"], "owner": "PLAYER_OPPONENT", "types": ["CREATURE"], "count": 1}
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

        # Process pending effects (ON_PLAY bounce)
        # Expect TARGET_SELECT -> RESOLVE_EFFECT
        for _ in range(5):
            actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, card_db_defs)
            next_act = None
            for a in actions:
                if a.type in [dm_ai_module.ActionType.RESOLVE_EFFECT, dm_ai_module.ActionType.SELECT_TARGET]:
                    next_act = a
                    break

            if next_act:
                 dm_ai_module.EffectResolver.resolve_action(state, next_act, card_db_defs)
            else:
                 break

        # Verify
        active = state.players[active_pid]
        opponent = state.players[opponent_pid]

        source_in_bz = False
        for c in active.battle_zone:
            if c.instance_id == 600: source_in_bz = True
        assert source_in_bz, "Source should be in battle zone"

        target_in_bz = False
        for c in opponent.battle_zone:
            if c.instance_id == 700: target_in_bz = True
        assert not target_in_bz, "Target should be returned to hand"

        if os.path.exists(json_path):
            os.remove(json_path)
