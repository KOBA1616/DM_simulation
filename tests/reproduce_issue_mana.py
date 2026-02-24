
import sys
import os
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../bin"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../build"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import dm_ai_module
from dm_toolkit import command_builders as cb

def test_mana_charge_repro():
    is_native = getattr(dm_ai_module, 'IS_NATIVE', False)
    if not is_native:
        print("WARNING: Not running with NATIVE module.")
    else:
        print("Running with NATIVE module.")

    # 1. Load Cards
    print("Loading cards...")
    try:
        cards_path = os.path.join(os.getcwd(), "data/cards.json")
        with open(cards_path, 'r', encoding='utf-8') as f:
            json_str = f.read()

        registry = getattr(dm_ai_module, "dm::engine::infrastructure::CardRegistry")
        registry.load_from_json(json_str)
        db = registry.get_all_cards()
        valid_card_id = list(db.keys())[0]
        print(f"Loaded cards. Using ID {valid_card_id}")

    except Exception as e:
        print(f"Failed to load cards: {e}")
        return

    # 2. Init Game
    gi = dm_ai_module.GameInstance(0, db)

    # 3. Setup Decks
    deck_ids = [valid_card_id] * 40
    gi.state.set_deck(0, deck_ids)
    gi.state.set_deck(1, deck_ids)

    gi.start_game()
    state = gi.state

    max_steps = 100
    steps = 0

    while steps < max_steps:
        phase = int(state.current_phase)
        print(f"Step {steps}: Current Phase: {phase}")

        if phase == 2: # Mana
            print("In Mana Phase (2). Attempting Charge via Pipeline (apply_move)...")
            p1 = state.players[0]
            if len(p1.hand) > 0:
                card = p1.hand[0]
                print(f"Charging card {card.instance_id}...")

                # Construct command for apply_move
                cmd = cb.build_mana_charge_command(
                    source_instance_id=card.instance_id,
                    native=True,
                    owner_id=0
                )

                # Execute via Pipeline
                state.apply_move(cmd)

                # Verify
                mana_count = len(p1.mana_zone)
                print(f"Mana Count: {mana_count}")
                if mana_count > 0:
                    print("SUCCESS: Mana charged.")
                else:
                    print("FAILURE: Mana not charged.")

                print(f"Phase after charge: {state.current_phase}")

                print("Passing to Main...")
                pass_cmd = cb.build_pass_command(native=True)
                state.apply_move(pass_cmd)

                print(f"Phase after PASS: {state.current_phase}")

                if int(state.current_phase) == 3:
                     print("Transitioned to Main Phase.")
                else:
                     print("Stuck in Mana Phase or other?")
                return
            else:
                print("Hand empty!")
                break

        res = gi.step()
        if not res:
            print("Game Over or No Actions")
            break

        steps += 1

    print("Reached max steps.")

if __name__ == "__main__":
    test_mana_charge_repro()
