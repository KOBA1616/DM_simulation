import pytest
import glob
import os
import json
import dm_ai_module

# Locate scenario files
SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
SCENARIO_FILES = glob.glob(os.path.join(SCENARIO_DIR, "*.json"))

def load_scenario(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def card_db():
    # Load the card database once for all generated tests
    # Assuming run from root, but handle relative paths safely
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    db_path = os.path.join(root, "data/cards.json")
    if not os.path.exists(db_path):
        # Fallback for different CWD
        db_path = "data/cards.json"
    return dm_ai_module.JsonLoader.load_cards(db_path)

@pytest.mark.parametrize("scenario_path", SCENARIO_FILES)
def test_generated_scenario(scenario_path, card_db):
    """
    Runs a generated scenario test case.
    """
    data = load_scenario(scenario_path)

    # 1. Setup Game State
    game = dm_ai_module.GameInstance(42)
    gs = game.state

    setup = data.get("setup", {})

    # Force player turn if specified
    if "player_turn" in setup:
        try:
            gs.turn_player = setup["player_turn"]
            gs.active_player_id = setup["player_turn"]
        except:
            pass

    # Setup zones
    players = setup.get("players", [])
    for p_idx, p_data in enumerate(players):
        player = gs.players[p_idx]

        # Hand
        if "hand" in p_data:
            # Clear existing hand if any (though new game starts empty usually)
            if hasattr(player.hand, "clear"):
                player.hand.clear()
            else:
                while len(player.hand) > 0: player.hand.pop()

            for cid in p_data["hand"]:
                # Use GameState method to ensure IDs are generated
                gs.add_card_to_hand(p_idx, cid)

        # Mana
        if "mana_zone" in p_data:
             if hasattr(player.mana_zone, "clear"):
                player.mana_zone.clear()
             else:
                while len(player.mana_zone) > 0: player.mana_zone.pop()

             for cid in p_data["mana_zone"]:
                 gs.add_card_to_mana(p_idx, cid)

        # Battle Zone
        if "battle_zone" in p_data:
             if hasattr(player.battle_zone, "clear"):
                player.battle_zone.clear()
             else:
                while len(player.battle_zone) > 0: player.battle_zone.pop()

             for cid in p_data["battle_zone"]:
                 gs.add_test_card_to_battle(p_idx, cid, gs.get_next_instance_id())

        # Shield Zone - if needed, though add_test_card_to_shield might not be in the minimal stub?
        # Checked dm_ai_module.py: PlayerStub has shield_zone list, GameState doesn't have add_to_shield.
        # But we can access the list directly.
        if "shield_zone" in p_data:
             if hasattr(player.shield_zone, "clear"):
                player.shield_zone.clear()
             else:
                while len(player.shield_zone) > 0: player.shield_zone.pop()

             for cid in p_data["shield_zone"]:
                 c = dm_ai_module.CardStub(cid, gs.get_next_instance_id())
                 player.shield_zone.append(c)

    # 2. Execute Action
    action_data = data.get("action", {})
    action_type = action_data.get("type")

    if action_type == "MANA_CHARGE":
        p_idx = action_data.get("player_index", 0)
        c_idx = action_data.get("card_index", 0)

        player = gs.players[p_idx]
        if 0 <= c_idx < len(player.hand):
            card = player.hand[c_idx]

            # Construct an Action-like object
            class MockAction:
                def __init__(self):
                    self.type = dm_ai_module.ActionType.MANA_CHARGE
                    self.card_id = card.card_id
                    self.source_instance_id = card.instance_id
                    self.target_player = p_idx

            action = MockAction()

            # Execute
            game.execute_action(action)

    # Add other action types as needed for future generated tests

    # 3. Assertions
    expected = data.get("expected", {})

    if "mana_zone_count_p0" in expected:
        assert len(gs.players[0].mana_zone) == expected["mana_zone_count_p0"]

    if "hand_count_p0" in expected:
        assert len(gs.players[0].hand) == expected["hand_count_p0"]

    if "battle_zone_count_p0" in expected:
        assert len(gs.players[0].battle_zone) == expected["battle_zone_count_p0"]

    if "shield_zone_count_p0" in expected:
        assert len(gs.players[0].shield_zone) == expected["shield_zone_count_p0"]
