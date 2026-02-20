
import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch

# Helper to ensure clean import
def clean_imports():
    if 'dm_toolkit.engine.compat' in sys.modules:
        del sys.modules['dm_toolkit.engine.compat']
    if 'dm_toolkit.dm_ai_module' in sys.modules:
        del sys.modules['dm_toolkit.dm_ai_module']

class TestCardLoadingConsistency(unittest.TestCase):
    TEMP_FILE = "verify_cards_temp.json"

    def setUp(self):
        clean_imports()
        # Create a dummy card list json
        self.card_list_data = [
            {"id": 1, "name": "Card One", "civilization": "Fire"},
            {"id": 2, "name": "Card Two", "civilization": "Water"}
        ]
        with open(self.TEMP_FILE, 'w') as f:
            json.dump(self.card_list_data, f)

    def tearDown(self):
        if os.path.exists(self.TEMP_FILE):
            os.remove(self.TEMP_FILE)
        clean_imports()

    def test_load_cards_robust_dual_loading(self):
        """
        Verify that load_cards_robust:
        1. Loads the JSON file as a Python dict (converting list to dict).
        2. Calls the Native loader to cache the C++ object.
        3. Returns the Python dict.
        """
        mock_native = MagicMock()
        mock_native_db = MagicMock(name="NativeDB")
        mock_native.JsonLoader.load_cards.return_value = mock_native_db
        mock_native.Phase = MagicMock()

        with patch.dict(sys.modules, {'dm_ai_module': mock_native, 'dm_toolkit.dm_ai_module': mock_native}):
            from dm_toolkit.engine.compat import EngineCompat

            EngineCompat.set_native_enabled(True)
            self.assertTrue(EngineCompat.is_available())

            loaded_db = EngineCompat.load_cards_robust(self.TEMP_FILE)

            self.assertIsInstance(loaded_db, dict)
            self.assertEqual(len(loaded_db), 2)
            self.assertEqual(loaded_db[1]['name'], "Card One")

            mock_native.JsonLoader.load_cards.assert_called_with(self.TEMP_FILE)
            self.assertEqual(EngineCompat._native_db_cache, mock_native_db)

            resolved = EngineCompat._resolve_db(loaded_db)
            self.assertEqual(resolved, mock_native_db)

    def test_load_cards_robust_native_disabled(self):
        """Verify behavior when native loading is disabled."""
        mock_native = MagicMock()

        with patch.dict(sys.modules, {'dm_ai_module': mock_native, 'dm_toolkit.dm_ai_module': mock_native}):
            from dm_toolkit.engine.compat import EngineCompat

            EngineCompat.set_native_enabled(False)

            loaded_db = EngineCompat.load_cards_robust(self.TEMP_FILE)

            self.assertIsInstance(loaded_db, dict)
            self.assertEqual(loaded_db[1]['name'], "Card One")

            mock_native.JsonLoader.load_cards.assert_not_called()
            self.assertIsNone(EngineCompat._native_db_cache)

            resolved = EngineCompat._resolve_db(loaded_db)
            self.assertEqual(resolved, loaded_db)

    def test_load_cards_robust_file_not_found(self):
        """Verify handling of missing files (both fail)."""
        missing_file = "non_existent_file_12345.json"

        mock_native = MagicMock()
        # Simulate native loader also failing to find file
        mock_native.JsonLoader.load_cards.return_value = None

        with patch.dict(sys.modules, {'dm_ai_module': mock_native, 'dm_toolkit.dm_ai_module': mock_native}):
            from dm_toolkit.engine.compat import EngineCompat

            loaded_db = EngineCompat.load_cards_robust(missing_file)

            # Should return empty dict
            self.assertEqual(loaded_db, {})

            # Native loader should still be attempted (since python failed)
            mock_native.JsonLoader.load_cards.assert_called()

    def test_load_cards_robust_fallback_to_native(self):
        """Verify fallback to native if python load fails but native succeeds."""
        # This could happen if file format is weird or just to test the path
        # We simulate python load failure by using a file that exists but raises exception or just forcing it
        # Easier to just use a non-existent file path but mock native to succeed (e.g. loading from memory or internal path)

        missing_file = "virtual_file.json"

        mock_native = MagicMock()
        mock_native_db = MagicMock(name="FallbackDB")
        mock_native.JsonLoader.load_cards.return_value = mock_native_db

        with patch.dict(sys.modules, {'dm_ai_module': mock_native, 'dm_toolkit.dm_ai_module': mock_native}):
            from dm_toolkit.engine.compat import EngineCompat

            # Python load will fail because file doesn't exist on disk
            loaded_db = EngineCompat.load_cards_robust(missing_file)

            # Should return the native db
            self.assertEqual(loaded_db, mock_native_db)

    def test_app_integration_check_list_handling(self):
        """
        Verify that the list-to-dict conversion happens inside load_cards_robust.
        """
        mock_native = MagicMock()
        with patch.dict(sys.modules, {'dm_ai_module': mock_native, 'dm_toolkit.dm_ai_module': mock_native}):
            from dm_toolkit.engine.compat import EngineCompat

            loaded_db = EngineCompat.load_cards_robust(self.TEMP_FILE)
            self.assertIsInstance(loaded_db, dict)
            self.assertTrue(1 in loaded_db)

if __name__ == '__main__':
    unittest.main()
