from dm_toolkit.gui.editor.models.serializer import ModelSerializer


def test_save_fails_on_duplicate_cost_reduction_ids(tmp_path):
    serializer = ModelSerializer()
    cards = [
        {
            "id": "c1",
            "name": "DupCard",
            "cost": 3,
            "cost_reductions": [
                {"type": "PASSIVE", "id": "dup"},
                {"type": "PASSIVE", "id": "dup"},
            ],
        }
    ]
    out = tmp_path / "out.json"
    ok = serializer.save_full_data(cards, str(out))
    assert ok is False
    assert not out.exists()


def test_save_succeeds_when_no_duplicate_ids(tmp_path):
    serializer = ModelSerializer()
    cards = [
        {
            "id": "c2",
            "name": "GoodCard",
            "cost": 2,
            "cost_reductions": [
                {"type": "PASSIVE", "id": "a"},
                {"type": "PASSIVE", "id": "b"},
            ],
        }
    ]
    out = tmp_path / "out.json"
    ok = serializer.save_full_data(cards, str(out))
    assert ok is True
    assert out.exists()
