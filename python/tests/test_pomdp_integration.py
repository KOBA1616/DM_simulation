import dm_ai_module


def test_parametric_belief_integration():
    b = dm_ai_module.ParametricBelief()
    # initialize with three card ids
    b.initialize_ids([1,2,3])

    vec0 = b.get_vector()
    assert len(vec0) == 3
    # sum to ~1
    assert abs(sum(vec0) - 1.0) < 1e-6

    state = dm_ai_module.GameState(0)
    # simulate opponent (player 1) having card id 1 in hand
    ci = dm_ai_module.CardInstance(1, 123)
    state.players[1].hand.append(ci)

    b.update(state)
    vec1 = b.get_vector()
    assert len(vec1) == 3
    # probabilities should still sum to ~1
    assert abs(sum(vec1) - 1.0) < 1e-6
    # probability for id 1 should decrease compared to before update
    assert vec1[0] < vec0[0]
