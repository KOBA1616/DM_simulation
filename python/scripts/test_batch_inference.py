import dm_ai_module as dm


def simple_model(batch_features):
    # batch_features: list of lists (float). Return (policies_list, values_list)
    # For testing, return empty policy vectors and zero values.
    policies = [[] for _ in batch_features]
    values = [0.0 for _ in batch_features]
    return policies, values


def main():
    print("has_reg before:", dm.has_batch_inference_registered())
    dm.register_batch_inference(simple_model)
    print("has_reg after:", dm.has_batch_inference_registered())

    # Prepare a simple GameState and call NeuralEvaluator
    gs = dm.GameState(0)
    gs.setup_test_duel()

    # NeuralEvaluator accepts a dict for card_db (we pass empty dict for test)
    ne = dm.NeuralEvaluator({})
    policies, values = ne.evaluate([gs])

    print("policies len:", len(policies))
    print("values len:", len(values))
    if policies:
        print("first policy len:", len(policies[0]))


if __name__ == '__main__':
    main()
