import os
import fnmatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def find_occurrences(pattern):
    matches = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for filename in filenames:
            if filename.endswith('.py'):
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        if pattern in text:
                            matches.append(path)
                except Exception:
                    pass
    return matches


def test_migration_boundary_single_layer():
    # We expect convert_legacy_action to be defined in transforms/legacy_to_command.py
    # and referenced only from models/serializer.py as the single migration boundary.
    occurrences = find_occurrences('convert_legacy_action')
    # Normalize paths to repository-relative
    rel = [os.path.relpath(p, ROOT).replace('\\', '/') for p in occurrences]
    # Expect exactly one reference (models/serializer.py) and the definition file
    assert any('transforms/legacy_to_command.py' in p for p in rel), f"Definition not found in transforms: {rel}"
    # Allow the definition and the serializer usage, but no other usages
    allowed = set(['dm_toolkit/gui/editor/transforms/legacy_to_command.py', 'dm_toolkit/gui/editor/models/serializer.py'])
    extras = [p for p in rel if p not in allowed]
    assert not extras, f"Unexpected references to convert_legacy_action found: {extras}"
