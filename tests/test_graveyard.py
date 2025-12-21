import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import dm_ai_module

def test_graveyard():
    print("Testing Graveyard Mechanics...")
    try:
        db_path = os.path.join(os.path.dirname(__file__), '../../data/cards.csv')
        card_db = dm_ai_module.CsvLoader.load_cards(db_path)

        gs = dm_ai_module.GameState(123)
        gs.setup_test_duel()

        # Manually add a creature to battle zone for P0 and P1
        # Assuming ID 1 is a creature
        p0 = gs.players[0]
        p1 = gs.players[1]

        # Add creature to P0 battle zone
        # c1 = dm_ai_module.CardInstance(1, 100)
        # c1.summoning_sickness = False
        # p0.battle_zone.append(c1)
        gs.add_test_card_to_battle(0, 1, 100, False, False) # P0, ID 1, Inst 100, Untapped, No Sick

        # Add creature to P1 battle zone
        # c2 = dm_ai_module.CardInstance(2, 101) # ID 2 is 2000 Power
        # c2.is_tapped = True
        # p1.battle_zone.append(c2)
        gs.add_test_card_to_battle(1, 2, 101, True, False) # P1, ID 2, Inst 101, Tapped, No Sick

        # Advance to Attack Phase
        dm_ai_module.PhaseManager.next_phase(gs, card_db) # START -> DRAW
        dm_ai_module.PhaseManager.next_phase(gs, card_db) # DRAW -> MANA
        dm_ai_module.PhaseManager.next_phase(gs, card_db) # MANA -> MAIN
        dm_ai_module.PhaseManager.next_phase(gs, card_db) # MAIN -> ATTACK
        print(f"Current Phase: {gs.current_phase}")

        # Refresh snapshots
        p0 = gs.players[0]
        p1 = gs.players[1]

        print(f"Initial P0 Battle Zone: {len(p0.battle_zone)}")
        print(f"Initial P0 Graveyard: {len(p0.graveyard)}")

        # Create Action: P0 attacks P1 creature (101) with creature (100)
        # ActionType.ATTACK_CREATURE = 4 (Check types.hpp or enum)
        # We need to know the enum value or use the module enum

        # Let's use ActionGenerator to find the action to be safe
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
        attack_action = None
        for a in actions:
            if a.type == dm_ai_module.ActionType.ATTACK_CREATURE and a.source_instance_id == 100 and a.target_instance_id == 101:
                attack_action = a
                break

        if not attack_action:
            print("Could not find attack action. Maybe summoning sickness?")
            # Clear summoning sickness
            # p0.battle_zone[0].summoning_sickness = False # Can't modify copy
            # We already set it to False in add_test_card_to_battle

            actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs, card_db)
            for a in actions:
                if a.type == dm_ai_module.ActionType.ATTACK_CREATURE and a.source_instance_id == 100 and a.target_instance_id == 101:
                    attack_action = a
                    break

        if attack_action:
            print("Executing Attack...")
            dm_ai_module.EffectResolver.resolve_action(gs, attack_action, card_db)

            # Now we should be in BLOCK phase or Battle resolved?
            # If no blockers, battle resolves immediately?
            # Let's check phase.
            print(f"Current Phase: {gs.current_phase}")

            if gs.current_phase == dm_ai_module.Phase.BLOCK:
                print("In Block Phase. Passing for P1 (No Block)...")
                pass_action = dm_ai_module.Action()
                pass_action.type = dm_ai_module.ActionType.PASS
                dm_ai_module.EffectResolver.resolve_action(gs, pass_action, card_db)

            # Refresh snapshots
            p0 = gs.players[0]

            # Now battle should be resolved.
            # P0 creature (1000) vs P1 creature (2000). P0 should die.
            print(f"Final P0 Battle Zone: {len(p0.battle_zone)}")
            print(f"Final P0 Graveyard: {len(p0.graveyard)}")

            if len(p0.graveyard) == 1 and p0.graveyard[0].instance_id == 100:
                print("SUCCESS: Creature moved to graveyard.")
            else:
                print("FAILURE: Creature did not move to graveyard.")
        else:
            print("FAILURE: Could not generate attack action.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_graveyard()
