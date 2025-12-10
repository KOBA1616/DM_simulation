import json
import logging
from typing import Dict, List, Set, Optional

class DeckClassifier:
    def __init__(self, meta_decks_path: str):
        self.meta_decks_path = meta_decks_path
        self.decks = []
        self._load_meta_decks()

    def _load_meta_decks(self):
        try:
            with open(self.meta_decks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.decks = data.get('decks', [])
        except Exception as e:
            logging.error(f"Failed to load meta decks from {self.meta_decks_path}: {e}")
            self.decks = []

    def classify(self, observed_cards: Set[int]) -> Dict[str, float]:
        """
        Classifies the deck based on observed card IDs.
        Returns a dictionary mapping deck names to probabilities (0.0 to 1.0).
        """
        if not self.decks:
            return {}

        scores = {}
        total_score = 0.0

        for deck in self.decks:
            name = deck['name']
            deck_cards = deck.get('cards', {})
            key_cards = set(deck.get('key_cards', []))

            # Simple scoring: +1 for each card match, +2 for key card match
            score = 0
            for card_id in observed_cards:
                str_id = str(card_id)
                if str_id in deck_cards:
                    score += 1
                    if card_id in key_cards:
                        score += 2 # Boost for key cards

            # Add a small base score to avoid zero probabilities if needed,
            # or just keep it 0 if no match.
            # Let's add a tiny epsilon to allow all decks to be possible initially?
            # For now, let's just stick to the score.

            scores[name] = score
            total_score += score

        # Normalize to probabilities
        probabilities = {}
        if total_score > 0:
            for name, score in scores.items():
                probabilities[name] = score / total_score
        else:
            # If no info, uniform distribution
            uniform_prob = 1.0 / len(self.decks)
            for deck in self.decks:
                probabilities[deck['name']] = uniform_prob

        return probabilities
