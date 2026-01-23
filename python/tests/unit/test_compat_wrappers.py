import unittest
from dm_toolkit.compat_wrappers import (
    add_aliases_to_command,
    is_legacy_command,
    normalize_legacy_fields,
    get_effective_command_type
)

class TestCompatWrappers(unittest.TestCase):

    def test_add_aliases_to_command(self):
        # 1. Basic usage
        cmd = {"type": "NEW_TYPE"}
        add_aliases_to_command(cmd, "OLD_TYPE")
        self.assertEqual(cmd.get("legacy_original_type"), "OLD_TYPE")
        self.assertIn("OLD_TYPE", cmd.get("aliases", []))

        # 2. Existing aliases
        cmd = {"type": "NEW_TYPE", "aliases": ["ANOTHER_OLD"]}
        add_aliases_to_command(cmd, "OLD_TYPE")
        self.assertIn("OLD_TYPE", cmd["aliases"])
        self.assertIn("ANOTHER_OLD", cmd["aliases"])

        # 3. Reason field
        cmd = {"type": "NEW_TYPE", "reason": "SPECIAL_REASON"}
        add_aliases_to_command(cmd, "OLD_TYPE")
        self.assertIn("SPECIAL_REASON", cmd["aliases"])

        # 4. Idempotency (avoid duplicates in list, order not guaranteed so use set)
        add_aliases_to_command(cmd, "OLD_TYPE")
        self.assertEqual(cmd["aliases"].count("OLD_TYPE"), 1)

    def test_is_legacy_command(self):
        # 1. Legacy original type present
        self.assertTrue(is_legacy_command({"legacy_original_type": "OLD"}))

        # 2. Aliases present
        self.assertTrue(is_legacy_command({"aliases": ["OLD"]}))

        # 3. Legacy warning present
        self.assertTrue(is_legacy_command({"legacy_warning": True}))

        # 4. Clean command
        self.assertFalse(is_legacy_command({"type": "NEW"}))

        # 5. Non-dict input
        self.assertFalse(is_legacy_command(None))

    def test_normalize_legacy_fields(self):
        # 1. str_param -> str_val
        cmd = {"str_param": "test"}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["str_val"], "test")

        # 2. str_val -> str_param
        cmd = {"str_val": "test"}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["str_param"], "test")

        # 3. amount -> value1
        cmd = {"amount": 10}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["value1"], 10)

        # 4. value1 -> amount
        cmd = {"value1": 10}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["amount"], 10)

        # 5. Both present (no change)
        cmd = {"amount": 10, "value1": 20}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["amount"], 10)
        self.assertEqual(cmd["value1"], 20)

        # 6. Both present string (no change)
        cmd = {"str_param": "new", "str_val": "old"}
        normalize_legacy_fields(cmd)
        self.assertEqual(cmd["str_param"], "new")
        self.assertEqual(cmd["str_val"], "old")

        # 7. Non-dict input
        self.assertIsNone(normalize_legacy_fields(None))

    def test_get_effective_command_type(self):
        # 1. Standard command
        cmd = {"type": "VALID_TYPE"}
        self.assertEqual(get_effective_command_type(cmd), "VALID_TYPE")

        # 2. NONE type without legacy info
        cmd = {"type": "NONE"}
        self.assertEqual(get_effective_command_type(cmd), "NONE")

        # 3. NONE type with legacy original type
        cmd = {"type": "NONE", "legacy_original_type": "OLD_TYPE"}
        self.assertEqual(get_effective_command_type(cmd), "OLD_TYPE")

        # 4. Valid type BUT legacy warning is set
        cmd = {"type": "WRONG_GUESS", "legacy_warning": True, "legacy_original_type": "REAL_TYPE"}
        self.assertEqual(get_effective_command_type(cmd), "REAL_TYPE")

        # 5. Missing type key
        cmd = {"legacy_original_type": "OLD_TYPE"}
        self.assertEqual(get_effective_command_type(cmd), "OLD_TYPE")

        # 6. Non-dict input
        self.assertIsNone(get_effective_command_type(None))
