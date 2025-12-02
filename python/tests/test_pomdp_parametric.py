import dm_ai_module


def test_parametric_belief_basic():
    b = dm_ai_module.ParametricBelief()
    # initialize with empty card_db -> should not throw
    b.initialize({})

    s = dm_ai_module.GameState(0)
    # update should accept state
    b.update(s)

    vec = b.get_vector()
    assert isinstance(vec, (list, tuple))
