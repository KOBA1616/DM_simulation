import dm_ai_module


def test_transition_reveal_penalty():
    # baseline belief
    b0 = dm_ai_module.ParametricBelief()
    b0.initialize_ids([1, 2, 3])
    vec0 = b0.get_vector()

    # prev state: card 1 is only in deck
    prev = dm_ai_module.GameState(0)
    ci_prev = dm_ai_module.CardInstance(1, 201)
    prev.players[1].deck.append(ci_prev)

    # curr state: card 1 moved to hand (a reveal)
    curr = dm_ai_module.GameState(0)
    ci_curr = dm_ai_module.CardInstance(1, 201)
    curr.players[1].hand.append(ci_curr)

    b_reveal = dm_ai_module.ParametricBelief()
    b_reveal.initialize_ids([1, 2, 3])
    # make reveal weight comparatively large
    b_reveal.set_weights(1.0, 0.1)
    b_reveal.set_reveal_weight(5.0)

    # update with prev->curr transition
    b_reveal.update_with_prev(prev, curr)
    v_reveal = b_reveal.get_vector()

    # Compare to updating with only curr (appearance in hand) or only deck listing
    b_deck_only = dm_ai_module.ParametricBelief()
    b_deck_only.initialize_ids([1, 2, 3])
    b_deck_only.set_weights(1.0, 0.1)
    b_deck_only.update(prev)  # deck listing only
    v_deck = b_deck_only.get_vector()

    # reveal penalty should be stronger than deck-only penalty
    assert v_reveal[0] < v_deck[0]
