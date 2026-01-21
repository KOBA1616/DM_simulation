import sys
import os
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dm_ai_module
from training.ai_player import AIPlayer

class TestAIPlayer:
    def test_ai_player_initialization(self):
        """Test that AIPlayer initializes correctly even without a model file."""
        player = AIPlayer("non_existent_model.pth")
        assert player.model is not None

    def test_ai_player_get_action(self):
        """Test that AIPlayer returns a valid GameCommand."""
        player = AIPlayer("non_existent_model.pth")

        # Setup mock game state
        gs = dm_ai_module.GameState()
        gs.players = [dm_ai_module.PlayerStub(), dm_ai_module.PlayerStub()]
        gs.active_player_id = 0
        gs.players[0].hand = [dm_ai_module.CardStub(1, 100)]
        gs.players[0].shield_zone = [dm_ai_module.CardStub(1, 200)]

        # Get action
        action = player.get_action(gs)

        # Check type
        assert isinstance(action, dm_ai_module.GameCommand)
        assert hasattr(action, 'type')
        # We can't predict the exact action with random weights, but it should be valid
        assert isinstance(action.type, int) or isinstance(action.type, dm_ai_module.ActionType)

    def test_decoder_logic(self):
        """Test the decoder mapping manually."""
        player = AIPlayer("non_existent_model.pth")
        gs = dm_ai_module.GameState()
        gs.players = [dm_ai_module.PlayerStub(), dm_ai_module.PlayerStub()]
        gs.active_player_id = 0

        # Add card to hand
        card = dm_ai_module.CardStub(card_id=1, instance_id=999)
        gs.players[0].hand.append(card)

        # Test Decode 0 -> PASS
        cmd_pass = player.decoder.decode(0, gs)
        assert cmd_pass.type == dm_ai_module.ActionType.PASS

        # Test Decode 1 -> MANA_CHARGE (index 0)
        cmd_mana = player.decoder.decode(1, gs)
        assert cmd_mana.type == dm_ai_module.ActionType.MANA_CHARGE
        assert cmd_mana.source_instance_id == 999

        # Test Decode 100 -> Out of bounds -> PASS
        cmd_invalid = player.decoder.decode(100, gs)
        assert cmd_invalid.type == dm_ai_module.ActionType.PASS

if __name__ == "__main__":
    pytest.main([__file__])
