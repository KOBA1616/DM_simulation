import dm_ai_module


def sample_model(batch):
    print('PY: sample_model called, batch length=', len(batch))
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    policies = [[0.0] * action_size for _ in batch]
    values = [0.0 for _ in batch]
    print('PY: sample_model returning')
    return policies, values


print('PY: registering sample model')
dm_ai_module.set_batch_callback(sample_model)
print('PY: registered?', dm_ai_module.has_batch_callback())

# Create simple state
s = dm_ai_module.GameState(0)
s.setup_test_duel()

ne = dm_ai_module.NeuralEvaluator({})
print('PY: about to call evaluate')
try:
    policies, values = ne.evaluate([s])
    print('PY: evaluate returned, lens:', len(policies), len(values))
    if policies:
        print('PY: policy0_len=', len(policies[0]))
except Exception as e:
    print('PY: exception:', e)
except BaseException as e:
    print('PY: BaseException:', type(e), e)
finally:
    dm_ai_module.clear_batch_callback()

print('PY: finished')
