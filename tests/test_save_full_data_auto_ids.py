import os
import json
from dm_toolkit.gui.editor.models.serializer import ModelSerializer


def test_save_full_data_assigns_ids(tmp_path):
    serializer = ModelSerializer()
    # Provide a list of card dicts missing cost_reduction ids
    cards = [
        {"id": "testcard", "name": "T", "cost": 3, "cost_reductions": [{"type": "PASSIVE"}]}
    ]
    out = tmp_path / "out.json"
    ok = serializer.save_full_data(cards, str(out))
    assert ok is True
    data = json.loads(out.read_text(encoding='utf-8'))
    assert isinstance(data, list) and len(data) == 1
    crs = data[0].get('cost_reductions')
    assert crs and isinstance(crs, list)
    assert 'id' in crs[0] and isinstance(crs[0]['id'], str) and crs[0]['id']
