import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from dm_toolkit.training.evolution_ecosystem import EvolutionEcosystem

# Use ParallelRunner via ecosystem convenience

def run_swap_test(games=100, cards_path=os.path.join('data','cards.json'), meta_path=os.path.join('data','meta_decks.json')):
    eco = EvolutionEcosystem(cards_path, meta_path)
    # use first available meta deck or simple repeated card id list
    card_db = eco.card_db
    # pick a simple deck from meta if available
    decks = None
    try:
        decks = eco.load_sample_decks(2)
    except Exception:
        # fallback: pick first card id repeated
        all_ids = list(card_db.keys())[:10]
        decks = [all_ids * 4, all_ids * 4]

    deck1 = decks[0]
    deck2 = decks[1]

    print('Running A vs B')
    res_ab = eco.runner.play_deck_matchup(deck1, deck2, games, 1)
    counts_ab = {0:res_ab.count(0), 1:res_ab.count(1), 2:res_ab.count(2)}
    print('A vs B counts:', counts_ab)

    print('Running B vs A')
    res_ba = eco.runner.play_deck_matchup(deck2, deck1, games, 1)
    counts_ba = {0:res_ba.count(0), 1:res_ba.count(1), 2:res_ba.count(2)}
    print('B vs A counts:', counts_ba)

    return counts_ab, counts_ba

if __name__ == '__main__':
    run_swap_test(200)
