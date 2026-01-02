
import pytest
import dm_ai_module
from dm_toolkit.action_to_command import map_action

class TestPhase6Schema:
    def test_zone_normalization(self):
        """Phase 6.1: Verify zone names are normalized to short enum forms."""

        # Test Case 1: MANA_CHARGE
        # Current policy: Keep specific CommandType for robustness (Phase A decision)
        action = {'type': 'MANA_CHARGE', 'from_zone': 'HAND', 'to_zone': 'MANA_ZONE', 'source_instance_id': 1}
        cmd = map_action(action)

        assert cmd['type'] == 'MANA_CHARGE', "MANA_CHARGE should preserve its type for engine compatibility"
        assert cmd['to_zone'] == 'MANA', "MANA_ZONE should be normalized to MANA"
        assert cmd['from_zone'] == 'HAND', "HAND should remain HAND"

        # Test Case 2: BATTLE_ZONE -> BATTLE
        action = {'type': 'PLAY_FROM_ZONE', 'from_zone': 'HAND', 'to_zone': 'BATTLE_ZONE', 'source_instance_id': 1}
        cmd = map_action(action)
        assert cmd['to_zone'] == 'BATTLE', "BATTLE_ZONE should be normalized to BATTLE"

        # Test Case 3: SHIELD_ZONE -> SHIELD
        action = {'type': 'ADD_SHIELD', 'from_zone': 'DECK', 'to_zone': 'SHIELD_ZONE'}
        cmd = map_action(action)
        assert cmd['to_zone'] == 'SHIELD', "SHIELD_ZONE should be normalized to SHIELD"

    def test_command_type_normalization(self):
        """Phase 6.2: Verify command types match implementation stability (Specific Types)."""

        # MANA_CHARGE
        action = {'type': 'MANA_CHARGE', 'from_zone': 'HAND', 'source_instance_id': 1}
        cmd = map_action(action)
        assert cmd['type'] == 'MANA_CHARGE'
        assert cmd['to_zone'] == 'MANA'

        # DESTROY
        action = {'type': 'DESTROY', 'source_instance_id': 1, 'from_zone': 'BATTLE'}
        cmd = map_action(action)
        assert cmd['type'] == 'DESTROY'
        assert cmd['to_zone'] == 'GRAVEYARD'

        # DRAW_CARD
        action = {'type': 'DRAW_CARD', 'source_instance_id': 1}
        cmd = map_action(action)
        assert cmd['type'] == 'DRAW_CARD'
        assert cmd['from_zone'] == 'DECK'
        assert cmd['to_zone'] == 'HAND'

    def test_play_card_mapping(self):
        """Verify PLAY_FROM_ZONE mapping."""
        action = {'type': 'PLAY_FROM_ZONE', 'from_zone': 'HAND', 'to_zone': 'BATTLE', 'source_instance_id': 1}
        cmd = map_action(action)
        assert cmd['type'] == 'PLAY_FROM_ZONE'
        assert cmd['to_zone'] == 'BATTLE'

    def test_attack_mapping(self):
        """Verify ATTACK_PLAYER mapping."""
        action = {'type': 'ATTACK_PLAYER', 'source_instance_id': 1, 'target_player': 1}
        cmd = map_action(action)
        assert cmd['type'] == 'ATTACK_PLAYER'
        assert cmd['target_player'] == 1
        assert cmd['instance_id'] == 1
