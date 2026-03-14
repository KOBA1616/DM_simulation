from pathlib import Path


def count_occurrences(text: str, sub: str) -> int:
    return text.count(sub)


def test_forms_safe_connect_coverage():
    root = Path('dm_toolkit/gui/editor/forms')
    py_files = list(root.rglob('*.py'))

    total_safe = 0
    total_raw = 0
    for p in py_files:
        text = p.read_text(encoding='utf-8')
        total_safe += count_occurrences(text, 'safe_connect(')
        total_raw += count_occurrences(text, '.connect(')

    # If there are no connect-like points, consider it passing
    total_points = total_safe + total_raw
    if total_points == 0:
        return

    coverage = total_safe / total_points

    assert coverage >= 0.8, f"safe_connect coverage too low: {coverage:.2%} ({total_safe}/{total_points})"
