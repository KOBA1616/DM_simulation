
import sys
import os
import random
import time

# Ensure we can import dm_ai_module from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dm_ai_module
except ImportError:
    # Fallback if running from tests/ directory directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import dm_ai_module

from dm_ai_module import GameInstance, ActionType, CommandType, CardType

def run_simulation(max_turns=100):
    print("Initializing Game Instance...")
    game = GameInstance()
    game.start_game()

    # Setup Default Decks (Simic Aggro - IDs from meta_decks.json)
    # IDs: 1, 7, 9, 12 repeated.
    deck_ids = [1, 7, 9, 12] * 10

    print("Setting up decks...")
    game.state.set_deck(0, deck_ids)
    game.state.set_deck(1, deck_ids)

    # Shuffle (Mock shuffle by randomizing the list in the stub if needed,
    # but the stub might not support shuffle. We can just proceed.)

    # Initial Draw (5 cards)
    print("Initial Draw...")
    game.state.draw_cards(0, 5)
    game.state.draw_cards(1, 5)

    # Setup Shields (5 cards)
    # The stub doesn't automatically do this in setup_test_duel usually.
    # We need to manually move cards from deck to shield zone.
    for pid in [0, 1]:
        for _ in range(5):
            if game.state.players[pid].deck:
                card = game.state.players[pid].deck.pop()
                game.state.players[pid].shield_zone.append(card)

    turn_count = 0
    while turn_count < max_turns:
        game.state.turn_number = turn_count // 2 + 1
        active_player = turn_count % 2
        game.state.active_player_id = active_player

        print(f"\n--- Turn {game.state.turn_number} (Player {active_player}) ---")

        # 1. Start Phase / Untap
        print("Phase: Untap")
        for card in game.state.players[active_player].battle_zone:
            card.is_tapped = False
            card.sick = False  # Clear summoning sickness
        for card in game.state.players[active_player].mana_zone:
            card.is_tapped = False

        # 2. Draw Phase
        print("Phase: Draw")
        game.state.draw_cards(active_player, 1)

        # 3. Mana Phase
        print("Phase: Mana Charge")
        hand = game.state.players[active_player].hand
        mana_count = len(game.state.players[active_player].mana_zone)

        # AI: Only charge if we have < 5 mana or plenty of cards
        if hand and (mana_count < 5 or len(hand) > 3):
            # Simple AI: Charge the highest cost card or random
            card_to_charge = hand[0] # Naive
            # Construct Action
            action = dm_ai_module.Action()
            action.type = ActionType.MANA_CHARGE
            action.source_instance_id = card_to_charge.instance_id
            action.card_id = card_to_charge.card_id
            action.target_player = active_player

            print(f"Player {active_player} charges card ID {card_to_charge.card_id}")
            game.execute_action(action)

            # Engine now handles MANA_CHARGE movement.

        # 4. Main Phase
        print("Phase: Main")
        # Attempt to play cards
        mana_zone = game.state.players[active_player].mana_zone
        mana_available = len([c for c in mana_zone if not c.is_tapped])
        hand = game.state.players[active_player].hand

        # print(f"DEBUG: Hand Size: {len(hand)}, Mana: {mana_available}/{len(mana_zone)}")
        # for c in hand:
        #      cost = get_card_cost(c.card_id)
        #      print(f"DEBUG: Card {c.card_id} Cost: {cost}")

        # Simple AI: Play first playable card
        playable_cards = [c for c in hand if get_card_cost(c.card_id) <= mana_available]

        if playable_cards:
            card_to_play = playable_cards[0]
            print(f"Player {active_player} plays card ID {card_to_play.card_id}")

            action = dm_ai_module.Action()
            action.type = ActionType.PLAY_CARD
            action.source_instance_id = card_to_play.instance_id
            action.card_id = card_to_play.card_id
            action.target_player = active_player

            # Execute
            game.execute_action(action)

            # Tap mana
            cost = get_card_cost(card_to_play.card_id)
            untapped_mana = [c for c in game.state.players[active_player].mana_zone if not c.is_tapped]
            for i in range(min(len(untapped_mana), cost)):
                untapped_mana[i].is_tapped = True

        # 5. Attack Phase
        print("Phase: Attack")
        # Simple AI: Attack Player with all eligible creatures
        attackers = [c for c in game.state.players[active_player].battle_zone if not c.is_tapped and not c.sick]

        for attacker in attackers:
            print(f"Player {active_player} attacks with instance {attacker.instance_id}")
            action = dm_ai_module.Action()
            action.type = ActionType.ATTACK_PLAYER
            action.source_instance_id = attacker.instance_id
            action.target_player = 1 - active_player
            game.execute_action(action)

            # Engine handles tapping.

            # Resolve Attack (Stub doesn't do this automatically yet, we do it here)
            opponent = 1 - active_player
            if game.state.players[opponent].shield_zone:
                # Break Shield
                shield = game.state.players[opponent].shield_zone.pop(0)
                game.state.players[opponent].hand.append(shield)
                print(f"Shield broken! Opponent shields: {len(game.state.players[opponent].shield_zone)}")
                # Check Shield Trigger?
            else:
                # Direct Attack
                print(f"Direct Attack! Player {active_player} WINS!")
                return active_player

        turn_count += 1

    print("Max turns reached. Draw.")
    return -1

def get_card_cost(card_id):
    # Mock cost lookup. In real engine, we'd query CardDatabase.
    # IDs: 1 (2), 7 (3), 9 (3), 12 (3)
    costs = {1: 2, 7: 3, 9: 3, 12: 3}
    return costs.get(card_id, 99)

if __name__ == "__main__":
    try:
        winner = run_simulation()
        if winner != -1:
            print(f"Simulation completed. Winner: Player {winner}")
            sys.exit(0)
        else:
            print("Simulation ended in Draw/Timeout.")
            sys.exit(1)
    except Exception as e:
        print(f"Simulation crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
