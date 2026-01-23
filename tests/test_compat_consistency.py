# -*- coding: utf-8 -*-
import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dm_toolkit.engine.compat import EngineCompat
import dm_ai_module

class TestEngineCompatConsistency(unittest.TestCase):

    def test_get_current_phase_without_enum(self):
        """Verify get_current_phase handles integer phases when Phase enum is missing."""
        state = dm_ai_module.GameState()
        # Mocking dm_ai_module to ensure Phase is not present if it happens to be
        with patch('dm_toolkit.engine.compat.dm_ai_module', new=dm_ai_module):
             # Force removal of Phase if it exists (it shouldn't in stub)
            if hasattr(dm_ai_module, 'Phase'):
                delattr(dm_ai_module, 'Phase')

            state.current_phase = 2 # Mana Phase
            phase = EngineCompat.get_current_phase(state)
            self.assertEqual(phase, 2)

            state.current_phase = "Phase.MANA"
            phase = EngineCompat.get_current_phase(state)
            self.assertEqual(phase, "Phase.MANA")

    def test_execute_command_fallback(self):
        """Verify ExecuteCommand falls back to Python implementation."""
        state = dm_ai_module.GameState()
        state.players[0].hand.append(dm_ai_module.CardStub(1, 100)) # ID 1, Instance 100

        # Command dict for MANA_CHARGE
        cmd = {
            'type': 'MANA_CHARGE',
            'instance_id': 100,
            'player_id': 0
        }

        # Verify initial state
        self.assertEqual(len(state.players[0].hand), 1)
        self.assertEqual(len(state.players[0].mana_zone), 0)

        # Execute
        EngineCompat.ExecuteCommand(state, cmd)

        # Verify result (should have moved from hand to mana)
        self.assertEqual(len(state.players[0].hand), 0)
        self.assertEqual(len(state.players[0].mana_zone), 1)
        self.assertEqual(state.players[0].mana_zone[0].instance_id, 100)

    def test_phase_manager_next_phase_fallback(self):
        """Verify PhaseManager_next_phase wraps the stub correctly."""
        state = dm_ai_module.GameState()
        state.current_phase = 2

        EngineCompat.PhaseManager_next_phase(state, {})

        # Stub logic: 2 -> 3
        self.assertEqual(state.current_phase, 3)

        EngineCompat.PhaseManager_next_phase(state, {})
        self.assertEqual(state.current_phase, 4)

    def test_phase_manager_force_advance(self):
        """Verify robustness: force advance if native call fails to change phase."""
        state = dm_ai_module.GameState()
        state.current_phase = 2

        # Mock dm_ai_module.PhaseManager.next_phase to do nothing
        with patch('dm_ai_module.PhaseManager.next_phase', side_effect=lambda s, d: None):
            # Ensure Phase is NOT present (simulating stub)
            # dm_ai_module.Phase does not exist, so we don't need to patch it out.

            # This SHOULD raise RuntimeError currently because force advance is skipped without Phase enum.
            # We want to eventually assert that it DOES NOT raise and advances phase.
            try:
                EngineCompat.PhaseManager_next_phase(state, {})
            except RuntimeError:
                self.fail("EngineCompat.PhaseManager_next_phase raised RuntimeError instead of forcing advance")

            # If it didn't raise, it should have advanced to 3 (or whatever next is)
            # But since we don't know "next" without enum, we expect it to increment if we implement integer fallback.
            self.assertNotEqual(state.current_phase, 2)

    def test_load_cards_robust(self):
        """Verify load_cards_robust prefers Python loader."""
        # Create a dummy json file
        import json
        dummy_cards = [{"id": 999, "name": "Test Card"}]
        filename = "test_cards.json"
        with open(filename, "w") as f:
            json.dump(dummy_cards, f)

        try:
            cards = EngineCompat.load_cards_robust(filename)
            self.assertIn(999, cards)
            self.assertEqual(cards[999]['name'], "Test Card")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == '__main__':
    unittest.main()
