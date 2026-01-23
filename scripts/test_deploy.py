from pathlib import Path
import sys
repo_root = Path.cwd()
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'bin' / 'Release'))
import dm_ai_module as dm
from training.deploy_with_onnx import main as deploy_main

models = list(Path('models').glob('duel_transformer_*.onnx'))
print('onnx count', len(models))
if models:
    onnx = str(sorted(models, key=lambda p:p.stat().st_mtime)[-1])
    print('using', onnx)
    try:
        deploy_main(onnx)
    except Exception as e:
        print('deploy_main raised:', e)
else:
    print('no onnx')
