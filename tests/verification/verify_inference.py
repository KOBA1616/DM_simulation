
import sys
import os
import json
import pytest

# Add bin/ to sys.path to find dm_ai_module
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found, skipping inference tests", allow_module_level=True)

class TestInference:
    def setup_method(self):
        # Create a dummy meta_decks.json
        self.meta_decks_path = "test_meta_decks.json"
        self.decks_data = {
            "decks": [
                {
                    "name": "AggroDeck",
                    "cards": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3] # 10 cards
                },
                {
                    "name": "ControlDeck",
                    "cards": [4, 4, 4, 4, 5, 5, 5, 5, 6, 6] # 10 cards
                }
            ]
        }
        with open(self.meta_decks_path, 'w') as f:
            json.dump(self.decks_data, f)

        # Create dummy card DB
        # We can't pass a raw python dict to a function expecting std::map<CardID, CardDefinition> unless exposed
        # But we can assume the engine loads one, or we can use the one from bindings if available.
        # But for test purposes, we need to pass it to PimcGenerator constructor.

        # We can use dm_ai_module.load_cards if we have a file, or create manually.
        # But load_cards returns void (updates internal registry or returns map?).
        # Actually `JsonLoader.load_cards` updates the global registry or returns?
        # Let's check JsonLoader binding or just rely on a dummy map if we can create CardDefinition.

        # Since we can't easily create C++ map in Python, we might be limited.
        # However, PimcGenerator binding takes `const std::map...&`. Pybind11 automatically converts dict to map.
        # So we just need to create CardDefinition objects.

        self.card_db = {}
        for i in range(1, 10):
            # CardDefinition binding?
            # It seems CardDefinition is bound but we need to see constructor.
            # Assuming default constructor works and fields are settable.
            c = dm_ai_module.CardDefinition()
            c.name = f"Card_{i}"
            c.cost = 1
            c.civilizations = [dm_ai_module.Civilization.FIRE]
            c.type = dm_ai_module.CardType.CREATURE
            self.card_db[i] = c

    def teardown_method(self):
        if os.path.exists(self.meta_decks_path):
            os.remove(self.meta_decks_path)

    def test_deck_inference_binding(self):
        """
        Verify that DeckInference binding works and can load decks.
        """
        if not hasattr(dm_ai_module, "DeckInference"):
            pytest.skip("DeckInference not exposed in dm_ai_module")

        inference = dm_ai_module.DeckInference()
        inference.load_decks(self.meta_decks_path)

        state = dm_ai_module.GameState(20) # 20 cards

        # Add card 1 (Aggro) to mana
        state.add_card_to_mana(1, 1, 100) # Player 1, Card 1, Inst 100

        # Infer from Player 0's perspective
        probs = inference.infer_probabilities(state, 0)

        assert "AggroDeck" in probs
        assert probs["AggroDeck"] > 0.9
        assert probs["ControlDeck"] < 0.1

    def test_pimc_generator_binding(self):
        """
        Verify PimcGenerator determinization.
        """
        if not hasattr(dm_ai_module, "PimcGenerator"):
            pytest.skip("PimcGenerator not exposed in dm_ai_module")

        inference = dm_ai_module.DeckInference()
        inference.load_decks(self.meta_decks_path)

        generator = dm_ai_module.PimcGenerator(self.card_db)
        generator.set_inference_model(inference)

        state = dm_ai_module.GameState(20) # 20 cards

        # Setup opponent (Player 1) with hidden cards (ID 0)
        # We need to manually add cards with ID 0.
        # Helper: add_card_to_hand(pid, cid, iid)

        # Add a visible card (ID 1 - Aggro)
        state.add_card_to_mana(1, 1, 100)

        # Add hidden cards to hand (ID 0)
        state.add_card_to_hand(1, 0, 101)
        state.add_card_to_hand(1, 0, 102)

        # Determinize
        # Seed 42
        det_state = generator.generate_determinized_state(state, 0, 42)

        # Check opponent hand
        opp = det_state.players[1]
        for card in opp.hand:
            assert card.card_id != 0
            # Should be from AggroDeck (1, 2, 3)
            assert card.card_id in [1, 2, 3]

if __name__ == "__main__":
    pytest.main([__file__])
