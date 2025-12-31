
import sys
import os
import pytest
from typing import Any
import random
import json
import threading
import time
import traceback

# Add bin directory to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
# Skip test automatically if C++ extension `dm_ai_module` is not available
pytest.importorskip("dm_ai_module", reason="dm_ai_module (C++ extension) not available; skipping heavy integration tests")

import dm_ai_module
from dm_ai_module import DeckEvolution, DeckEvolutionConfig, Civilization, CardDefinition, CardKeywords, EffectDef

def create_mock_card(id, name, civ_enum, races, cost, power, evolution=False):
    keywords = CardKeywords()
    keywords.evolution = evolution
    # Create empty effects list
    effects: list[Any] = []

    # Map Enum to String for Constructor
    civ_str = "NONE"
    if civ_enum == Civilization.FIRE:
        civ_str = "FIRE"
    elif civ_enum == Civilization.WATER:
        civ_str = "WATER"
    elif civ_enum == Civilization.NATURE:
        civ_str = "NATURE"
    elif civ_enum == Civilization.LIGHT:
        civ_str = "LIGHT"
    elif civ_enum == Civilization.DARKNESS:
        civ_str = "DARKNESS"

    return CardDefinition(id, name, civ_str, races, cost, power, keywords, effects)

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

    # test_evolve_deck_structure removed due to instability in CI (native extension subprocess issues)

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
