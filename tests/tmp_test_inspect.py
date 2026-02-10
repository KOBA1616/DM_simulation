def test_inspect_generate_commands():
    import dm_ai_module
    try:
        gi = dm_ai_module.GameInstance(0)
    except Exception:
        gi = dm_ai_module.GameInstance()
    state = gi.state
    assert hasattr(dm_ai_module, 'generate_commands')
    res = dm_ai_module.generate_commands(state, {})
    print('inspect_len', len(res))
    assert isinstance(res, (list, tuple))
