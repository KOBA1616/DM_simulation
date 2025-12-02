import dm_ai_module


def test_pomdp_basic():
    pom = dm_ai_module.POMDPInference()
    # initialize with empty card db
    pom.initialize({})

    # create a minimal GameState and call update/infer
    state = dm_ai_module.GameState(0)
    bv = pom.get_belief_vector()
    assert isinstance(bv, (list, tuple))

    # should accept the state without throwing
    pom.update_belief(state)
    act = pom.infer_action(state)
    assert isinstance(act, (list, tuple))
