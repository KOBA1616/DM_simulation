
import pytest
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

class TestTextGeneratorExtended:
    def test_zone_fallback_unknown_pair(self):
        """Test fallback for zone pair that doesn't have a specific natural language template."""
        # SHIELD_ZONE -> DECK is not explicitly handled in the big if-elif block for TRANSITION
        cmd = {
            "type": "TRANSITION",
            "from_zone": "SHIELD_ZONE",
            "to_zone": "DECK",
            "target_filter": {"zones": ["SHIELD_ZONE"]}, # Implicitly defines target
            "amount": 1
        }
        # Expected fallback: "{target}を{from_z}から{to_z}へ移動する。"
        # localized: "シールドをシールドゾーンから山札へ移動する。"
        # Note: _resolve_target might return "シールド" or "カード" depending on filter.
        # Let's see what happens.

        generated = CardTextGenerator._format_command(cmd)

        # We expect a generic movement message because no specific "Shield to Deck" template exists in _format_zone_move_command
        # It should at least be readable.
        assert "移動する" in generated
        assert "シールドゾーン" in generated or "シールド" in generated
        assert "山札" in generated or "デッキ" in generated

    def test_zone_fallback_weird_zones(self):
        """Test fallback with completely custom/unknown zones."""
        cmd = {
            "type": "TRANSITION",
            "from_zone": "DIMENSION_ZONE",
            "to_zone": "VOID",
            "amount": 1
        }

        generated = CardTextGenerator._format_command(cmd)

        # Should rely on tr() fallback which usually returns the key if not found,
        # or just use the key if no translation.
        # Expected: "カードをDIMENSION_ZONEからVOIDへ移動する。" (or similar)
        assert "移動する" in generated
        assert "DIMENSION_ZONE" in generated
        assert "VOID" in generated

    def test_summon_token_undefined(self):
        """Test SUMMON_TOKEN with an undefined/untranslated token ID."""
        cmd = {
            "type": "SUMMON_TOKEN",
            "str_val": "UNKNOWN_TOKEN_ID_999",
            "value1": 2
        }

        generated = CardTextGenerator._format_command(cmd)

        # Expect fallback to generic "トークン" since ID is internal-looking
        assert "トークン" in generated
        assert "UNKNOWN_TOKEN_ID_999" not in generated
        assert "2体出す" in generated

    def test_summon_token_empty(self):
        """Test SUMMON_TOKEN with empty ID."""
        cmd = {
            "type": "SUMMON_TOKEN",
            "str_val": "",
            "value1": 1
        }

        generated = CardTextGenerator._format_command(cmd)

        assert "トークンを1体出す" in generated

    def test_transition_deck_to_shield(self):
         """Test specific unusual transition: Deck to Shield (not as Shield Trigger processing, but direct move)."""
         # Usually "ADD_SHIELD" is used, but if TRANSITION is used:
         cmd = {
             "type": "TRANSITION",
             "from_zone": "DECK",
             "to_zone": "SHIELD_ZONE",
             "amount": 1
         }

         generated = CardTextGenerator._format_command(cmd)
         # Expect fallback or generic
         assert "山札" in generated
         assert "シールドゾーン" in generated
         assert "移動する" in generated
