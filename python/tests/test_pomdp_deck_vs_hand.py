import dm_ai_module


def test_deck_vs_hand_penalty():
    b = dm_ai_module.ParametricBelief()
    b.initialize_ids([1,2,3])
    # baseline
    vec0 = b.get_vector()

    s_deck = dm_ai_module.GameState(0)
    # simulate card id 1 present in deck listing
    ci = dm_ai_module.CardInstance(1, 101)
    s_deck.players[1].deck.append(ci)

    b_deck = dm_ai_module.ParametricBelief()
    b_deck.initialize_ids([1,2,3])
    b_deck.update(s_deck)
    v_deck = b_deck.get_vector()

    s_hand = dm_ai_module.GameState(0)
    ci2 = dm_ai_module.CardInstance(1, 102)
    s_hand.players[1].hand.append(ci2)

    b_hand = dm_ai_module.ParametricBelief()
    b_hand.initialize_ids([1,2,3])
    b_hand.update(s_hand)
    v_hand = b_hand.get_vector()

    # both should remain normalized
    assert abs(sum(v_deck) - 1.0) < 1e-6
    assert abs(sum(v_hand) - 1.0) < 1e-6

    # penalty for hand should be stronger than deck: probability for id 1 should be no greater in hand case
    assert v_hand[0] <= v_deck[0]
