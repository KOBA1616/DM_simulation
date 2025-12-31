
import pytest
import dm_ai_module
from dm_ai_module import CardDefinition, CommandType, CommandDef, TargetScope

# This test replaces test_keyword_effects.py by verifying "Command" generation instead of "Action" generation.
# This aligns with the "Test Rectification" requirement.

def test_mega_last_burst_command_generation():
    """Verify Mega Last Burst creates ON_DESTROY effect with CAST_SPELL (SELF) command"""

    # Create a minimal JSON snippet with Mega Last Burst
    json_data = """
    {
        "id": 1000,
        "name": "MLB Test",
        "type": "CREATURE",
        "keywords": {
            "mega_last_burst": true
        },
        "spell_side": {
            "id": 1001,
            "name": "MLB Spell",
            import pytest

            pytest.skip("obsolete legacy-action tests removed", allow_module_level=True)
                    "trigger": "NONE",
