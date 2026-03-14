from pathlib import Path
from tools.cir_cost import analyze_cir_cost


def test_analyze_cir_cost_keys_and_types():
    root = Path('.')
    metrics = analyze_cir_cost(root)
    expected_keys = {
        'normalize_refs', 'serializer_refs', 'data_manager_refs',
        'model_files', 'model_classes', 'python_files'
    }
    assert set(metrics.keys()) >= expected_keys
    for k in expected_keys:
        assert isinstance(metrics[k], int)
