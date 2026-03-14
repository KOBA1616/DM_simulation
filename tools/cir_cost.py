from pathlib import Path
from typing import Dict


def analyze_cir_cost(root: Path) -> Dict[str, int]:
    """Simple heuristic analysis of CIR adoption surface:
    - counts references to normalize, serializer, and data_manager
    - counts number of model classes in dm_toolkit/gui/editor/models
    Returns a dict of metrics.
    """
    metrics = {
        'normalize_refs': 0,
        'serializer_refs': 0,
        'data_manager_refs': 0,
        'model_files': 0,
        'model_classes': 0,
        'python_files': 0,
    }

    root = Path(root)
    for p in root.rglob('*.py'):
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        metrics['python_files'] += 1
        if 'normalize' in text:
            metrics['normalize_refs'] += text.count('normalize')
        if 'serializer' in text:
            metrics['serializer_refs'] += text.count('serializer')
        if 'data_manager' in text:
            metrics['data_manager_refs'] += text.count('data_manager')

    models_dir = root / 'dm_toolkit' / 'gui' / 'editor' / 'models'
    if models_dir.exists():
        for p in models_dir.rglob('*.py'):
            metrics['model_files'] += 1
            try:
                text = p.read_text(encoding='utf-8')
            except Exception:
                continue
            # crude class count
            metrics['model_classes'] += text.count('\nclass ')

    return metrics


if __name__ == '__main__':
    import json
    from pathlib import Path

    root = Path('.')
    m = analyze_cir_cost(root)
    print(json.dumps(m, indent=2))
