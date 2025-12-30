import importlib.machinery
import importlib.util
import sys
import os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
pyd_path = os.path.join(root, 'bin', 'dm_ai_module.cp312-win_amd64.pyd')

def load_native():
    loader = importlib.machinery.ExtensionFileLoader('dm_ai_module', pyd_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod

def main():
    print('pyd_path =', pyd_path)
    if not os.path.exists(pyd_path):
        print('pyd not found')
        return
    try:
        mod = load_native()
        print('loaded:', getattr(mod, '__file__', None))

        print('calling set_batch_callback')
        def sample(batch):
            return ([[0.1] * getattr(mod.ActionEncoder, 'TOTAL_ACTION_SIZE', 10) for _ in batch], [0.5 for _ in batch])
        mod.set_batch_callback(sample)
        print('has_batch_callback ->', getattr(mod, 'has_batch_callback', lambda: False)())

        print('creating GameState')
        s = mod.GameState(0)
        print('setup_test_duel')
        s.setup_test_duel()

        print('instantiating NeuralEvaluator')
        ne = mod.NeuralEvaluator({})

        print('calling evaluate')
        p, v = ne.evaluate([s])
        print('evaluate returned', len(p), len(v))

    except Exception as e:
        print('exception:', type(e), e)

if __name__ == '__main__':
    main()
