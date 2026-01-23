
import unittest
import sys
import os
import uuid
import copy
from unittest.mock import patch, MagicMock

from dm_toolkit.action_to_command import (
    map_action,
    normalize_action_zone_keys,
    _get_zone,
    _get_any,
    _transfer_targeting,
    _transfer_common_move_fields,
    _finalize_command,
    _validate_command_type,
    set_command_type_enum
)
from dm_toolkit import action_to_command

class TestHelpers(unittest.TestCase):

    def test_normalize_action_zone_keys(self):
        # Case 1: No legacy keys
        data = {"other": 1}
        self.assertEqual(normalize_action_zone_keys(data), data)

        # Case 2: from_zone -> source_zone
        data = {"from_zone": "A"}
        norm = normalize_action_zone_keys(data)
        self.assertEqual(norm["source_zone"], "A")
        self.assertEqual(norm["from_zone"], "A")

        # Case 3: to_zone -> destination_zone
        data = {"to_zone": "B"}
        norm = normalize_action_zone_keys(data)
        self.assertEqual(norm["destination_zone"], "B")
        self.assertEqual(norm["to_zone"], "B")

        # Case 4: canonical keys already exist (don't overwrite)
        data = {"from_zone": "A", "source_zone": "X"}
        norm = normalize_action_zone_keys(data)
        self.assertEqual(norm["source_zone"], "X")

        # Case 5: Not a dict
        self.assertIsNone(normalize_action_zone_keys(None))

    def test_get_zone(self):
        d = {"a": "1", "b": "2"}
        self.assertEqual(_get_zone(d, ["a", "b"]), "1")
        self.assertEqual(_get_zone(d, ["c", "b"]), "2")
        self.assertIsNone(_get_zone(d, ["c", "d"]))

    def test_get_any(self):
        d = {"a": 0, "b": 1}
        # 0 is a valid value, should be returned
        self.assertEqual(_get_any(d, ["a", "b"]), 0)
        self.assertEqual(_get_any(d, ["c", "b"]), 1)
        self.assertIsNone(_get_any(d, ["c"]))

    def test_transfer_targeting(self):
        cmd = {}
        # Case 1: Defaults
        _transfer_targeting({}, cmd)
        self.assertEqual(cmd.get('target_group'), 'NONE')

        # Case 2: Filter implies TARGET_SELECT
        act = {'filter': {'race': 'dragon'}}
        cmd = {}
        _transfer_targeting(act, cmd)
        self.assertEqual(cmd['target_group'], 'TARGET_SELECT')
        self.assertEqual(cmd['target_filter'], {'race': 'dragon'})

        # Case 3: Optional flag
        act = {'optional': True}
        cmd = {}
        _transfer_targeting(act, cmd)
        self.assertIn('OPTIONAL', cmd['flags'])

    def test_transfer_common_move_fields(self):
        # Case 1: amount
        cmd = {}
        _transfer_common_move_fields({'amount': 5}, cmd)
        self.assertEqual(cmd['amount'], 5)

        # Case 2: filter count
        cmd = {}
        _transfer_common_move_fields({'filter': {'count': 3}}, cmd)
        self.assertEqual(cmd['amount'], 3)

        # Case 3: value1
        cmd = {}
        _transfer_common_move_fields({'value1': 2}, cmd)
        self.assertEqual(cmd['amount'], 2)

    def test_finalize_command(self):
        # Case 1: UID generation
        cmd = {}
        _finalize_command(cmd, {})
        self.assertTrue('uid' in cmd)

        # Case 2: Amount normalization (value1 -> amount)
        cmd = {'value1': 10}
        _finalize_command(cmd, {})
        self.assertEqual(cmd['amount'], 10)

        # Case 3: Amount from action
        cmd = {}
        act = {'value1': 5}
        _finalize_command(cmd, act)
        self.assertEqual(cmd['amount'], 5)

        # Case 4: str_val -> str_param
        cmd = {}
        act = {'str_val': 'test'}
        _finalize_command(cmd, act)
        self.assertEqual(cmd['str_param'], 'test')

        # Case 5: value2 preservation
        cmd = {}
        act = {'value2': 99}
        _finalize_command(cmd, act)
        self.assertEqual(cmd['value2'], 99)

        # Case 6: Flags propagation
        cmd = {}
        act = {'flags': ['A']}
        _finalize_command(cmd, act)
        self.assertEqual(cmd['flags'], ['A'])

        # Case 7: Zero handling
        cmd = {}
        act = {'value1': 0}
        _finalize_command(cmd, act)
        self.assertEqual(cmd['amount'], 0) # Should be 0, not None

    def test_validate_command_type(self):
        # Mocking logic for _CommandType
        class MockEnum:
            TEST_CMD = 1

        # Inject mock
        set_command_type_enum(MockEnum)

        # Case 1: Valid type
        cmd = {'type': 'TEST_CMD'}
        _validate_command_type(cmd)
        self.assertFalse(cmd.get('legacy_warning'))

        # Case 2: Invalid type
        cmd = {'type': 'INVALID_CMD'}
        _validate_command_type(cmd)
        self.assertTrue(cmd.get('legacy_warning'))
        self.assertEqual(cmd['legacy_invalid_type'], 'INVALID_CMD')

        # Case 3: Virtual type allowed
        cmd = {'type': 'CHOICE'}
        _validate_command_type(cmd)
        self.assertFalse(cmd.get('legacy_warning'))

        # Reset
        set_command_type_enum(None)


class TestHandlers(unittest.TestCase):

    def test_handle_replace_card_move(self):
        act = {
            'type': 'REPLACE_CARD_MOVE',
            'source_zone': 'HAND',
            'source_instance_id': 10
        }
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'REPLACE_CARD_MOVE')
        self.assertEqual(cmd['current_zone'], 'HAND')
        self.assertEqual(cmd['to_zone'], 'DECK_BOTTOM') # Default
        self.assertEqual(cmd['instance_id'], 10)

        # Test with original_to_zone (simulating external wrapper logic)
        # map_action itself doesn't pull "original_to_zone" from act directly unless we add it to helper?
        # Actually _handle_replace_card_move takes args (act, cmd, original_zone, dest).
        # In map_action: `src = _get_zone(...)` which gets `source_zone`. `dest = ...` which gets `to_zone`.
        # Wait, `REPLACE_CARD_MOVE` usually implies we *intercepted* a move.
        # If input act has `to_zone`="GRAVEYARD", it maps to `dest`.
        act2 = {'type': 'REPLACE_CARD_MOVE', 'to_zone': 'GRAVEYARD'}
        cmd2 = map_action(act2)
        self.assertEqual(cmd2['to_zone'], 'GRAVEYARD')

        # Note: Logic says `if original_zone: cmd['original_to_zone'] = original_zone`.
        # `original_zone` comes from `src`. Wait.
        # Line 275: `src = _get_zone(...)`.
        # Line 326: `_handle_replace_card_move(act_data, cmd, src, dest)`
        # Inside handler: `if original_zone: cmd['original_to_zone'] = original_zone`
        # So if we provide `from_zone`="GRAVE", it becomes `original_to_zone`. This seems like a specific usage pattern.
        # Let's verify that.
        act3 = {'type': 'REPLACE_CARD_MOVE', 'from_zone': 'GRAVEYARD'}
        cmd3 = map_action(act3)
        self.assertEqual(cmd3['original_to_zone'], 'GRAVEYARD')


    def test_handle_move_card_mapping(self):
        # Generic MOVE_CARD
        act = {'type': 'MOVE_CARD', 'from_zone': 'A', 'to_zone': 'B'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'TRANSITION')

        # Implicit DESTROY
        act = {'type': 'MOVE_CARD', 'from_zone': 'BATTLE', 'to_zone': 'GRAVEYARD'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'DESTROY')

        # Implicit DISCARD
        act = {'type': 'MOVE_CARD', 'from_zone': 'HAND', 'to_zone': 'GRAVEYARD'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'DISCARD')

        # Implicit RETURN_TO_HAND
        act = {'type': 'MOVE_CARD', 'from_zone': 'BATTLE', 'to_zone': 'HAND'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'RETURN_TO_HAND')

    def test_handle_specific_moves(self):
        # MANA_CHARGE (Hand -> Mana)
        act = {'type': 'MANA_CHARGE'}
        cmd = map_action(act)
        self.assertEqual(cmd['to_zone'], 'MANA')
        self.assertEqual(cmd['from_zone'], 'HAND')

        # ADD_MANA (Deck -> Mana)
        act = {'type': 'ADD_MANA'}
        cmd = map_action(act)
        self.assertEqual(cmd['to_zone'], 'MANA')
        self.assertEqual(cmd['from_zone'], 'DECK')

        # SEARCH_DECK_BOTTOM
        act = {'type': 'SEARCH_DECK_BOTTOM'}
        cmd = map_action(act)
        self.assertEqual(cmd['to_zone'], 'DECK_BOTTOM')
        self.assertEqual(cmd['from_zone'], 'DECK')

        # ADD_SHIELD
        act = {'type': 'ADD_SHIELD'}
        cmd = map_action(act)
        self.assertEqual(cmd['to_zone'], 'SHIELD')
        self.assertEqual(cmd['from_zone'], 'DECK')

        # MOVE_TO_UNDER_CARD
        act = {'type': 'MOVE_TO_UNDER_CARD', 'base_target': 123}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'TRANSITION')
        self.assertEqual(cmd['to_zone'], 'UNDER_CARD')
        self.assertEqual(cmd['base_target'], 123)

    def test_handle_modifiers(self):
        # COST_REDUCTION
        act = {'type': 'COST_REDUCTION', 'value1': 1}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'MUTATE')
        self.assertEqual(cmd['mutation_kind'], 'COST')
        self.assertEqual(cmd['amount'], 1)

        # GRANT_KEYWORD
        act = {'type': 'GRANT_KEYWORD', 'str_val': 'SPEED_ATTACKER'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'ADD_KEYWORD')
        self.assertEqual(cmd['mutation_kind'], 'SPEED_ATTACKER')

    def test_handle_mutate(self):
        # POWER_MOD
        act = {'type': 'POWER_MOD', 'value1': 1000}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'MUTATE')
        self.assertEqual(cmd['mutation_kind'], 'POWER_MOD')
        self.assertEqual(cmd['amount'], 1000)

        # SET_POWER
        act = {'type': 'MUTATE', 'str_val': 'SET_POWER', 'value1': 5000}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'MUTATE')
        self.assertEqual(cmd['mutation_kind'], 'POWER_SET')
        self.assertEqual(cmd['amount'], 5000)

        # TAP/UNTAP via string param
        act = {'type': 'MUTATE', 'str_val': 'TAP'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'TAP')

    def test_handle_selection(self):
        # SELECT_OPTION
        act = {'type': 'SELECT_OPTION', 'value1': 2}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'CHOICE')
        self.assertEqual(cmd['amount'], 2)

        # SELECT_NUMBER
        act = {'type': 'SELECT_NUMBER', 'min': 1, 'max': 5}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'SELECT_NUMBER')
        self.assertEqual(cmd['min'], 1)
        self.assertEqual(cmd['max'], 5)

    def test_handle_complex(self):
        # SEARCH_DECK
        act = {'type': 'SEARCH_DECK', 'value1': 1}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'SEARCH_DECK')
        self.assertEqual(cmd['unified_type'], 'SEARCH')

        # MEKRAID
        act = {'type': 'MEKRAID', 'value1': 7, 'value2': 3}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'MEKRAID')
        self.assertEqual(cmd['max_cost'], 7)
        self.assertEqual(cmd['look_count'], 3)
        self.assertTrue(cmd['play_for_free'])

    def test_handle_play_flow(self):
        # PLAY_CARD -> PLAY_FROM_ZONE (Hand->Battle)
        act = {'type': 'PLAY_CARD', 'value1': 5}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'PLAY_FROM_ZONE')
        self.assertEqual(cmd['from_zone'], 'HAND')

        # PLAY_FROM_ZONE
        act = {'type': 'PLAY_FROM_ZONE', 'from_zone': 'MANA', 'play_for_free': True}
        cmd = map_action(act)
        self.assertEqual(cmd['from_zone'], 'MANA')
        self.assertIn('PLAY_FOR_FREE', cmd['play_flags'])

    def test_handle_engine_execution(self):
        # ATTACK_PLAYER with 0
        act = {'type': 'ATTACK_PLAYER', 'source_instance_id': 0, 'target_player': 1}
        cmd = map_action(act)
        self.assertEqual(cmd['instance_id'], 0)
        self.assertEqual(cmd['target_player'], 1)

        # BLOCK
        act = {'type': 'BLOCK', 'source_instance_id': 10, 'target_id': 20}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'FLOW')
        self.assertEqual(cmd['flow_type'], 'BLOCK')
        self.assertEqual(cmd['instance_id'], 10)
        self.assertEqual(cmd['target_instance'], 20)

    def test_handle_buffer_ops(self):
        # SELECT_FROM_BUFFER
        act = {'type': 'SELECT_FROM_BUFFER', 'value1': 1}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'SELECT_FROM_BUFFER')
        self.assertEqual(cmd['amount'], 1)

        # SUMMON_TOKEN
        act = {'type': 'SUMMON_TOKEN', 'str_val': 'T001'}
        cmd = map_action(act)
        self.assertEqual(cmd['token_id'], 'T001')

    def test_misc(self):
        # Invalid input shape
        cmd = map_action("NotADict")
        self.assertEqual(cmd['type'], 'NONE')
        self.assertIn('Invalid action shape', cmd['str_param'])

        # Uncopyable
        class BadObj:
            def to_dict(self): raise Exception("Fail")
        cmd = map_action(BadObj())
        self.assertEqual(cmd['type'], 'NONE')
        self.assertIn('Uncopyable', cmd['str_param'])

        # Legacy Keyword fallback
        act = {'type': 'NONE', 'str_val': 'MY_KEYWORD'}
        cmd = map_action(act)
        self.assertEqual(cmd['type'], 'ADD_KEYWORD')
        self.assertEqual(cmd['mutation_kind'], 'MY_KEYWORD')

if __name__ == '__main__':
    unittest.main()
