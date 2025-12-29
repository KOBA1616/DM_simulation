import importlib.machinery, importlib.util, sys, os, traceback
p = os.path.join(os.getcwd(),'bin','dm_ai_module.cp312-win_amd64.pyd')
print('p=',p)
loader = importlib.machinery.ExtensionFileLoader('dm_ai_module', p)
spec = importlib.util.spec_from_loader('dm_ai_module', loader)
mod = importlib.util.module_from_spec(spec)
try:
    loader.exec_module(mod)
    print('loaded, has NeuralEvaluator=', hasattr(mod,'NeuralEvaluator'))
except Exception:
    traceback.print_exc()
