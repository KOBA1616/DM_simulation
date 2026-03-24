import json
from dm_toolkit.gui.editor.models.serializer import ModelSerializer


def test_save_full_data_writes_list(tmp_path):
    ser = ModelSerializer()
    sample = [
        {"id": 1, "name": "Card A", "effects": [], "static_abilities": [], "reaction_abilities": []}
    ]
    out = tmp_path / "cards_out.json"
    ok = ser.save_full_data(sample, str(out))
    assert ok is True
    # verify file content
    data = json.loads(out.read_text(encoding='utf-8'))
    assert isinstance(data, list)
    assert data[0]["id"] == 1
