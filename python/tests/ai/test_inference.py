import unittest
import json
import os
from dm_toolkit.ai.inference.deck_classifier import DeckClassifier
from dm_toolkit.ai.inference.hand_estimator import HandEstimator

class TestInference(unittest.TestCase):
    def setUp(self):
        # Create a temporary meta_decks.json for testing
        self.test_decks_path = "test_meta_decks.json"
        self.decks_data = {
            "decks": [
                {
                    "name": "Nature Ramp",
                    "cards": {
                        "1": 4,
                        "5": 4
                    },
                    "key_cards": [1]
                },
                {
                    "name": "Fire Aggro",
                    "cards": {
                        "2": 4,
                        "3": 4
                    },
                    "key_cards": [2, 3]
                }
            ]
        }
        with open(self.test_decks_path, 'w', encoding='utf-8') as f:
            json.dump(self.decks_data, f, ensure_ascii=False)

    def tearDown(self):
        if os.path.exists(self.test_decks_path):
            os.remove(self.test_decks_path)

    def test_deck_classification(self):
        classifier = DeckClassifier(self.test_decks_path)

        # Test 1: Observation matching Nature Ramp
        probs = classifier.classify({1, 5})
        self.assertGreater(probs["Nature Ramp"], probs["Fire Aggro"])
        print(f"Observed {1, 5} -> {probs}")

        # Test 2: Observation matching Fire Aggro (Key Card 2)
        probs = classifier.classify({2})
        self.assertGreater(probs["Fire Aggro"], probs["Nature Ramp"])
        print(f"Observed {2} -> {probs}")

        # Test 3: Ambiguous or mixed (not really possible with disjoint sets here, but let's try just ID 3)
        probs = classifier.classify({3})
        self.assertGreater(probs["Fire Aggro"], probs["Nature Ramp"])

    def test_hand_estimation(self):
        classifier = DeckClassifier(self.test_decks_path)
        estimator = HandEstimator(classifier)

        # Observe ID 1 (Nature Ramp indicator)
        estimator.update([1])

        # Check estimation.
        # Deck should be heavily weighted to Nature Ramp.
        # ID 5 (Terror Pit) is in Nature Ramp, so its probability should be high.
        # ID 2 (Speed Attacker) is NOT in Nature Ramp, so its probability should be low.

        cards_played = {1: 1} # We saw 1 copy of ID 1
        probs = estimator.estimate_hand_cards(cards_played)

        print(f"Hand Probs after seeing ID 1: {probs}")

        # ID 5 should have non-zero probability
        self.assertGreater(probs.get(5, 0.0), 0.0)

        # ID 2 should be very low (ideally 0 if probability became 100% Nature Ramp,
        # but the classifier allows non-zero for others usually)
        # In our simple implementation:
        # Nature Ramp score: 1 (match) + 2 (key) = 3
        # Fire Aggro score: 0
        # Probabilities: Nature=1.0, Fire=0.0

        self.assertGreater(probs.get(5, 0.0), probs.get(2, 0.0))
        self.assertAlmostEqual(probs.get(2, 0.0), 0.0)

if __name__ == '__main__':
    unittest.main()
