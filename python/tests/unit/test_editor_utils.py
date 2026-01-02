
import pytest
from dm_toolkit.gui.editor.utils import normalize_action_zone_keys, normalize_command_zone_keys

def test_normalize_action_zone_keys():
    # Case 1: Standard input (no change)
    data = {"source_zone": "HAND", "destination_zone": "GRAVEYARD"}
    normalize_action_zone_keys(data)
    assert data["source_zone"] == "HAND"
    assert data["destination_zone"] == "GRAVEYARD"
    assert "from_zone" not in data
    assert "to_zone" not in data

    # Case 2: Legacy input (convert)
    data = {"from_zone": "HAND", "to_zone": "GRAVEYARD"}
    normalize_action_zone_keys(data)
    assert data["source_zone"] == "HAND"
    assert data["destination_zone"] == "GRAVEYARD"
    assert "from_zone" not in data
    assert "to_zone" not in data

    # Case 3: Mixed input (preserve source/dest)
    data = {"source_zone": "HAND", "to_zone": "GRAVEYARD"}
    normalize_action_zone_keys(data)
    assert data["source_zone"] == "HAND"
    assert data["destination_zone"] == "GRAVEYARD"
    assert "to_zone" not in data

    # Case 4: No zones
    data = {"type": "DESTROY"}
    normalize_action_zone_keys(data)
    assert "from_zone" not in data
    assert "to_zone" not in data
    assert "source_zone" not in data

def test_normalize_command_zone_keys():
    # Case 1: Standard input
    data = {"from_zone": "HAND", "to_zone": "GRAVEYARD"}
    normalize_command_zone_keys(data)
    assert data["from_zone"] == "HAND"
    assert data["to_zone"] == "GRAVEYARD"
    assert "source_zone" not in data
    assert "destination_zone" not in data

    # Case 2: Action-like input (convert)
    data = {"source_zone": "HAND", "destination_zone": "GRAVEYARD"}
    normalize_command_zone_keys(data)
    assert data["from_zone"] == "HAND"
    assert data["to_zone"] == "GRAVEYARD"
    assert "source_zone" not in data
    assert "destination_zone" not in data
