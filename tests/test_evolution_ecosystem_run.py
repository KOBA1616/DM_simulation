
import os
import sys
import pytest
import shutil
import json

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

try:
    import dm_ai_module
except ImportError:
    pass # Let the test fail if module missing in setup

# Import the ecosystem class
# Need to make sure dm_toolkit is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from dm_toolkit.training.evolution_ecosystem import EvolutionEcosystem

class TestEvolutionEcosystemRun:
    def setup_method(self):
        # Create temp meta file
        self.cards_path = "data/cards.json"
        self.meta_path = "data/temp_test_meta_decks.json"

        # Copy valid meta to temp if exists, else create valid
        if os.path.exists("data/meta_decks.json"):
            shutil.copy("data/meta_decks.json", self.meta_path)
        else:
             with open(self.meta_path, 'w') as f:
                 json.dump({"decks": []}, f)

    def teardown_method(self):
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

    def test_run_single_generation(self):
        if not os.path.exists(self.cards_path):
            pytest.skip("cards.json not found")

        ecosystem = EvolutionEcosystem(self.cards_path, self.meta_path)

        # Run 1 generation with 1 challenger, 2 games per match to be fast
        ecosystem.run_generation(num_challengers=1, games_per_match=2)

        # Verify file was touched/saved if accepted, or at least exists
        assert os.path.exists(self.meta_path)

        # Verify JSON validity
        with open(self.meta_path, 'r') as f:
            data = json.load(f)
            assert "decks" in data
            # assert len(data["decks"]) >= 0

if __name__ == "__main__":
    pytest.main([__file__])
