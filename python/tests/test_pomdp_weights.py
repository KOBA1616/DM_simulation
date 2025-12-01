import dm_ai_module


def test_weights_make_hand_stronger_than_deck():
    b = dm_ai_module.ParametricBelief()
    b.initialize_ids([1,2,3])

    # Set strong weight high and deck weight very small
    b.set_weights(5.0, 0.01)

    # baseline
    vec0 = b.get_vector()

    s_deck = dm_ai_module.GameState(0)
    ci = dm_ai_module.CardInstance(1, 101)
    s_deck.players[1].deck.append(ci)

    b_deck = dm_ai_module.ParametricBelief()
    b_deck.initialize_ids([1,2,3])
    b_deck.set_weights(5.0, 0.01)
    b_deck.update(s_deck)
    v_deck = b_deck.get_vector()

    s_hand = dm_ai_module.GameState(0)
    ci2 = dm_ai_module.CardInstance(1, 102)
    s_hand.players[1].hand.append(ci2)

    b_hand = dm_ai_module.ParametricBelief()
    b_hand.initialize_ids([1,2,3])
    b_hand.set_weights(5.0, 0.01)
    b_hand.update(s_hand)
    v_hand = b_hand.get_vector()

    # With strong weights configured strictly, hand penalty should be stronger than deck
    assert v_hand[0] < v_deck[0]
