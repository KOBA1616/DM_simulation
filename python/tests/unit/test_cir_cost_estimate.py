from tools.cir_cost import analyze_cir_cost
from pathlib import Path
import json


def test_analyze_cir_cost_returns_metrics_and_dump():
    root = Path('.')
    metrics = analyze_cir_cost(root)

    assert isinstance(metrics, dict)
    # basic expected keys
    for k in ['normalize_refs', 'serializer_refs', 'data_manager_refs', 'model_files', 'model_classes', 'python_files']:
        assert k in metrics
        assert isinstance(metrics[k], int)

    # write metrics to tools/cir_metrics.json for plan update
    out = Path('tools') / 'cir_metrics.json'
    out.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
