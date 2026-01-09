
import os
import json
import random
import copy
from typing import List, Dict, Any, Optional

try:
    import dm_ai_module
except ImportError:
    # If module is not found, we might be in a CI/Test env where it's not built yet
    # or path is not set.
    # For now, we assume it's available or we mock it in tests.
    dm_ai_module = None

class DeckIndividual:
    """
    Represents a single deck in the population.
    """
    def __init__(self, deck_id: str, cards: List[int], win_rate: float = 0.0):
        self.deck_id = deck_id
        self.cards = cards
        self.win_rate = win_rate
        self.matchups: Dict[str, float] = {} # opponent_deck_id -> win_rate
        self.games_played = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deck_id": self.deck_id,
            "cards": self.cards,
            "win_rate": self.win_rate,
            "matchups": self.matchups,
            "games_played": self.games_played
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DeckIndividual':
        deck = DeckIndividual(data["deck_id"], data["cards"], data.get("win_rate", 0.0))
        deck.matchups = data.get("matchups", {})
        deck.games_played = data.get("games_played", 0)
        return deck

class PopulationManager:
    """
    Manages the population of decks for the evolution ecosystem.
    Responsible for initialization, saving/loading, and basic population operations.
    """
    def __init__(self, card_db: Dict[int, Any], population_size: int = 20, storage_path: str = "data/population"):
        self.card_db = card_db
        self.population_size = population_size
        self.storage_path = storage_path
        self.population: List[DeckIndividual] = []
        self.generation = 0

        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)

    def initialize_random_population(self, candidate_pool_ids: List[int], deck_size: int = 40):
        """
        Initializes the population with random decks created from the candidate pool.
        """
        self.population = []
        self.generation = 0

        for i in range(self.population_size):
            deck_cards = []
            # Simple random selection for now
            # In future, we might want to respect civilization balance or mana curve
            for _ in range(deck_size):
                deck_cards.append(random.choice(candidate_pool_ids))

            deck_id = f"gen0_deck{i:03d}"
            self.population.append(DeckIndividual(deck_id, deck_cards))

    def load_population(self, filename: str = "current_population.json"):
        """
        Loads the population from a JSON file.
        """
        filepath = os.path.join(self.storage_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Population file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)
        self.population = [DeckIndividual.from_dict(d) for d in data.get("decks", [])]

    def save_population(self, filename: str = "current_population.json"):
        """
        Saves the current population to a JSON file.
        """
        data = {
            "generation": self.generation,
            "timestamp": "", # TODO: Add timestamp
            "decks": [d.to_dict() for d in self.population]
        }

        filepath = os.path.join(self.storage_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_population(self) -> List[DeckIndividual]:
        return self.population

    def update_generation(self, new_population: List[DeckIndividual]):
        """
        Updates the population with a new generation of decks.
        """
        self.population = new_population
        self.generation += 1

    def get_deck_by_id(self, deck_id: str) -> Optional[DeckIndividual]:
        for deck in self.population:
            if deck.deck_id == deck_id:
                return deck
        return None
