import dm_ai_module
import sys
import random

def verify_deck_evolution_logic():
    print("Verifying C++ Deck Evolution Logic...")

    # Mock data
    fixed_cards = [1, 2, 3] # Fixed cards (e.g. key cards)
    candidate_pool = [10, 11, 12, 13, 14, 15] # Available new cards
    current_deck = [1, 2, 3, 4, 5, 4, 5, 6, 7, 8] # Current deck

    # Create dummy card stats
    # Map<int, CardStats>
    card_stats = {}

    # Card 4: High Usage (Played 10 times, won 5 times) -> Should be kept
    s4 = dm_ai_module.CardStats()
    s4.play_count = 10
    s4.mana_source_count = 2
    s4.sum_win_contribution = 5.0 # 50% win rate
    card_stats[4] = s4

    # Card 5: High Resource Use (Played 0, Mana 20) -> Should have score ~2.0
    s5 = dm_ai_module.CardStats()
    s5.play_count = 0
    s5.mana_source_count = 20
    s5.sum_win_contribution = 0.0
    card_stats[5] = s5

    # Card 6: Low Usage (Played 0, Mana 0) -> Score 0 -> Should be removed
    s6 = dm_ai_module.CardStats()
    s6.play_count = 0
    s6.mana_source_count = 0
    s6.sum_win_contribution = 0.0
    card_stats[6] = s6

    # Card 7: Medium usage
    s7 = dm_ai_module.CardStats()
    s7.play_count = 2
    s7.mana_source_count = 2
    s7.sum_win_contribution = 1.0
    card_stats[7] = s7

    # Card 8: Low Usage -> Remove
    s8 = dm_ai_module.CardStats()
    s8.play_count = 0
    s8.mana_source_count = 1
    s8.sum_win_contribution = 0.0
    card_stats[8] = s8

    # Initialize Evolver
    evolver = dm_ai_module.DeckEvolution()
    # Default configs: active=5.0, resource=2.0, win_weight=1.0

    print("Checking score calculation...")
    score4 = evolver.calculate_score(s4)
    # interactions = 12. usage = 10*5 + 2*2 = 54. base = 54/12 = 4.5. win = 5/10 = 0.5. Total = 5.0
    print(f"Card 4 Score: {score4}")

    score5 = evolver.calculate_score(s5)
    # interactions = 20. usage = 0 + 20*2 = 40. base = 2.0. win = 0. Total = 2.0
    print(f"Card 5 Score: {score5}")

    score6 = evolver.calculate_score(s6)
    # 0
    print(f"Card 6 Score: {score6}")

    if score4 <= score5:
        print("FAIL: Card 4 should have higher score than Card 5")
        sys.exit(1)

    if score6 != 0.0:
        print("FAIL: Card 6 should have 0 score")
        sys.exit(1)

    print("Evolving deck...")
    # Removing 2 cards
    new_deck = evolver.evolve_deck(current_deck, card_stats, candidate_pool, fixed_cards, 2)

    print(f"Old Deck: {current_deck}")
    print(f"New Deck: {new_deck}")

    # Checks
    # Length should remain same (assuming evolve_deck replaces)
    # Wait, implementation: removes N, adds N.
    if len(new_deck) != len(current_deck):
        print("FAIL: Deck size changed")
        sys.exit(1)

    # Card 6 and 8 should likely be removed (scores 0 and ~2)
    # Card 5 (score 2) might be at risk compared to 4 and 7.
    # Scores: 4=5.0, 5=2.0, 6=0.0, 7=(10+4)/4 + 0.5 = 3.5 + 0.5 = 4.0, 8=(0+2)/1 + 0 = 2.0
    # Lowest are 6 (0.0), 8 (2.0), 5 (2.0).
    # Should remove 6 and one of 5 or 8.

    if 6 in new_deck:
        print("FAIL: Card 6 (Score 0) was not removed")
        sys.exit(1)

    # Fixed cards (1, 2, 3) must be present even if they have no stats (score 0)
    for c in fixed_cards:
        if c not in new_deck:
            print(f"FAIL: Fixed card {c} was removed")
            sys.exit(1)

    # New cards from candidate pool should be present
    added_count = 0
    for c in new_deck:
        if c in candidate_pool:
            added_count += 1

    if added_count != 2:
        print(f"FAIL: Expected 2 new cards from candidate pool, found {added_count}")
        sys.exit(1)

    print("Deck Evolution Verification Passed.")

if __name__ == "__main__":
    verify_deck_evolution_logic()
