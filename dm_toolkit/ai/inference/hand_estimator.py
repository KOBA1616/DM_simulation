from typing import Dict, List, Set, Optional
from .deck_classifier import DeckClassifier

class HandEstimator:
    def __init__(self, deck_classifier: DeckClassifier):
        self.classifier = deck_classifier
        self.observed_cards: Set[int] = set()
        self.deck_probabilities: Dict[str, float] = {}
        # Default fallback values if game state specifics aren't provided
        self.default_pool_size = 35.0
        self.default_hand_size = 5.0

    def update(self, observed_card_ids: List[int]):
        """
        Updates the internal state with newly observed card IDs (e.g., from mana zone, graveyard, battle zone).
        """
        for card_id in observed_card_ids:
            self.observed_cards.add(card_id)

        self.deck_probabilities = self.classifier.classify(self.observed_cards)

    def estimate_hand_cards(self, cards_played: Dict[int, int],
                            hand_size: Optional[int] = None,
                            unknown_pool_size: Optional[int] = None) -> Dict[int, float]:
        """
        Estimates the probability of each card ID being in the opponent's hand.

        Args:
            cards_played: Dictionary mapping card ID to count of copies already seen public zones.
            hand_size: The current number of cards in the opponent's hand. Defaults to 5.
            unknown_pool_size: The total number of unknown cards (Deck + Hand + Shields). Defaults to 35.

        Returns:
            Dictionary mapping card ID to probability (0.0 to 1.0).
        """
        current_hand_size = float(hand_size) if hand_size is not None else self.default_hand_size
        current_pool_size = float(unknown_pool_size) if unknown_pool_size is not None else self.default_pool_size

        card_probs = {}

        # 1. Aggregate likely composition of opponent's deck based on deck probabilities
        weighted_deck_composition: Dict[str, float] = {} # str(id) -> expected count

        if not self.deck_probabilities:
             # If no update called yet, initialize
             self.deck_probabilities = self.classifier.classify(self.observed_cards)

        for deck_def in self.classifier.decks:
            name = deck_def['name']
            prob = self.deck_probabilities.get(name, 0.0)

            for card_id_str, count in deck_def.get('cards', {}).items():
                current_val = weighted_deck_composition.get(card_id_str, 0.0)
                weighted_deck_composition[card_id_str] = current_val + (count * prob)

        for card_id_str, expected_total_count in weighted_deck_composition.items():
            card_id = int(card_id_str)
            seen_count = cards_played.get(card_id, 0)
            remaining_count = max(0.0, expected_total_count - seen_count)

            if current_pool_size > 0:
                # Simplified probability: (Remaining / Unknown) * HandSize
                # We clamp at 1.0
                prob = min(1.0, (remaining_count / current_pool_size) * current_hand_size)
                card_probs[card_id] = prob
            else:
                card_probs[card_id] = 0.0

        return card_probs
