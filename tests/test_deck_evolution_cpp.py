
import sys
import os
import pytest
import random
import json

# Add bin directory to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

try:
    import dm_ai_module
except ImportError:
    pytest.fail("Could not import dm_ai_module. Make sure the C++ module is built and in the bin directory.")

from dm_ai_module import DeckEvolution, DeckEvolutionConfig, Civilization, CardDefinition, CardKeywords, EffectDef

def create_mock_card(id, name, civ, races, cost, power, evolution=False):
    keywords = CardKeywords()
    keywords.evolution = evolution
    # Create empty effects list
    effects = []
    return CardDefinition(id, name, civ, races, cost, power, keywords, effects)

class TestDeckEvolutionCpp:
    @pytest.fixture
    def card_db(self):
        db = {}
        # 1. Fire Speed Attacker
        db[1] = create_mock_card(1, "Fire Bird", Civilization.FIRE, ["FireBird"], 3, 3000)
        # 2. Fire Dragon (Evolves from Fire Bird)
        db[2] = create_mock_card(2, "Fire Dragon", Civilization.FIRE, ["Armored Dragon"], 6, 6000, evolution=True)
        # 3. Water Blocker
        db[3] = create_mock_card(3, "Water Guard", Civilization.WATER, ["Liquid People"], 2, 2000)
        # 4. Nature Mana
        db[4] = create_mock_card(4, "Nature Elf", Civilization.NATURE, ["Snow Faerie"], 2, 1000)
        # 5. Evolution Target for Dragon (Same race to test synergy)
        db[5] = create_mock_card(5, "Little Dragon", Civilization.FIRE, ["Armored Dragon"], 3, 3000)
        return db

    def test_evolve_deck_structure(self, card_db):
        evolver = DeckEvolution(card_db)
        config = DeckEvolutionConfig()
        config.target_deck_size = 40
        config.mutation_rate = 0.5 # Force some mutations

        current_deck = [1] * 40
        candidate_pool = [2, 3, 4]

        new_deck = evolver.evolve_deck(current_deck, candidate_pool, config)

        assert len(new_deck) == 40
        assert any(c in candidate_pool for c in new_deck)
        # Should still contain some original cards
        assert any(c == 1 for c in new_deck)

    def test_interaction_score_synergy(self, card_db):
        evolver = DeckEvolution(card_db)

        # Deck 1: Fire Bird + Fire Dragon (Evolution synergy?)
        # Current logic: Evolution gets bonus if races match.
        # Fire Bird race: FireBird. Fire Dragon race: Armored Dragon.
        # No direct race match.
        # But same civ -> +1.0

        deck1 = [1, 2]
        score1 = evolver.calculate_interaction_score(deck1)

        # Deck 2: Fire Bird + Water Guard
        # Diff civ, diff race -> Score 0?
        deck2 = [1, 3]
        score2 = evolver.calculate_interaction_score(deck2)

        assert score1 > score2

    def test_evolution_race_synergy(self, card_db):
        evolver = DeckEvolution(card_db)

        # Deck 3: Little Dragon (Armored Dragon) + Fire Dragon (Armored Dragon, Evolution)
        # Match Civ (+1)
        # Match Race (+0.5)
        # Evolution Synergy (Evo + Race Match) (+2.0)
        # Expected high score
        deck3 = [5, 2]
        score3 = evolver.calculate_interaction_score(deck3)

        # Deck 1: Fire Bird + Fire Dragon
        # Match Civ (+1)
        # No Race match
        deck1 = [1, 2]
        score1 = evolver.calculate_interaction_score(deck1)

        assert score3 > score1

    def test_get_candidates_by_civ(self, card_db):
        evolver = DeckEvolution(card_db)
        pool = [1, 2, 3, 4]

        fire_cards = evolver.get_candidates_by_civ(pool, Civilization.FIRE)
        assert 1 in fire_cards
        assert 2 in fire_cards
        assert 3 not in fire_cards

        water_cards = evolver.get_candidates_by_civ(pool, Civilization.WATER)
        assert 3 in water_cards
        assert 1 not in water_cards

if __name__ == "__main__":
    pytest.main([__file__])
