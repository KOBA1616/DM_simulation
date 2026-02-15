def test_tensor_length():
    import dm_ai_module as dm
    from dm_ai_module import TensorConverter, GameState
    s = GameState(0)
    card_db = {}
    vec = TensorConverter.convert_to_tensor(s, s.active_player_id, card_db)
    assert len(vec) == dm.TensorConverter.INPUT_SIZE
