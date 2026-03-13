from dm_toolkit.gui.editor.models import CardModel


def test_keywords_model_loads_and_serializes_legacy_shape():
    # Legacy dict with flag and condition-like entries
    legacy = {
        'flying': True,
        'friend_burst_condition': {'when': 'ally_play'},
        'custom_tag': {'data': 123}
    }

    card = CardModel(keywords=legacy)

    # Keywords should become the structured KeywordsModel
    assert isinstance(card.keywords, CardModel.KeywordsModel)

    dumped = card.model_dump()
    assert 'keywords' in dumped
    kw = dumped['keywords']

    # Legacy-style fields must be present after serialization
    assert kw.get('flying') is True
    assert 'friend_burst_condition' in kw and kw['friend_burst_condition'] == {'when': 'ally_play'}
    # custom_tag should be preserved (as extras)
    assert 'custom_tag' in kw and kw['custom_tag'] == {'data': 123}
