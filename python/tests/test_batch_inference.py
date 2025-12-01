import dm_ai_module


def sample_model(batch):
    # batch: list[list[float]]
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    policies = [[0.0] * action_size for _ in batch]
    values = [0.0 for _ in batch]
    return policies, values


def run_test():
    print('registering sample model')
    dm_ai_module.register_batch_inference(sample_model)
    print('registered?', dm_ai_module.has_batch_inference_registered())

    # Create simple state
    s = dm_ai_module.GameState(0)
    s.setup_test_duel()

    # Instantiate NeuralEvaluator with empty card DB
    ne = dm_ai_module.NeuralEvaluator({})

    policies, values = ne.evaluate([s])
    print('policies_len=', len(policies), 'values_len=', len(values))
    if policies:
        print('policy0_len=', len(policies[0]))

    assert len(values) == 1
    assert len(policies) == 1


if __name__ == '__main__':
    run_test()
