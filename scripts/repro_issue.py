
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.DEBUG)

from dm_toolkit.types import GameState, CardDB, Action
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.game_session import GameSession
import dm_ai_module

def test_repro():
    print("--- Starting Repro ---")

    # 1. Setup Mock Card DB
    card_db = {
        1: {"id": 1, "name": "Test Creature", "type": "CREATURE", "cost": 2, "civilizations": ["FIRE"], "power": 1000},
        2: {"id": 2, "name": "Test Spell", "type": "SPELL", "cost": 1, "civilizations": ["WATER"]},
    }

    # 2. Initialize Game via Session (to test the binding logic)
    session = GameSession(
        callback_log=lambda x: print(f"[LOG] {x}"),
        callback_update_ui=lambda: None
    )

    # Manually inject deck to avoid random logic if possible
    session.DEFAULT_DECK = [1, 2, 1, 2, 1, 2, 1, 2] * 4

    print("Initializing Game...")
    session.initialize_game(card_db)

    # 3. Check Initial State
    gs = session.gs
    p0 = gs.players[0]
    print(f"P0 Hand: {len(p0.hand)}")
    print(f"P0 Mana: {len(p0.mana_zone)}")
    print(f"Phase: {gs.current_phase}")

    # 4. Generate Actions
    print("Generating Actions...")
    actions = session.generate_legal_actions()
    print(f"Legal Actions: {len(actions)}")
    for i, a in enumerate(actions):
        print(f" {i}: {a.to_dict()}")

    # 5. Attempt Mana Charge (if available)
    charge_action = None
    for a in actions:
        d = a.to_dict()
        if d.get('type') == 'MANA_CHARGE' or d.get('type') == 7: # 7 is MANA_CHARGE in int enum
            charge_action = a
            break

    if charge_action:
        print(f"Executing Mana Charge: {charge_action.to_dict()}")
        session.execute_action(charge_action)

        # 6. Verify Result
        print(f"P0 Hand After: {len(p0.hand)}")
        print(f"P0 Mana After: {len(p0.mana_zone)}")

        if len(p0.mana_zone) == 1:
            print("SUCCESS: Mana Charge executed.")
        else:
            print("FAILURE: Mana Zone did not increase.")
    else:
        print("FAILURE: No MANA_CHARGE action generated.")

if __name__ == "__main__":
    test_repro()
