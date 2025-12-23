import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replace Keyword:: with dm::core::Keyword::
    # But avoid double prefix if already qualified (dm::core::Keyword::)
    # Also avoid if inside dm::core namespace? No, safe to qualify.
    # Regex: (?<!dm::core::)Keyword::

    new_content = re.sub(r'(?<!dm::core::)Keyword::', r'dm::core::Keyword::', content)

    if new_content != content:
        print(f"Fixing {filepath}")
        with open(filepath, 'w') as f:
            f.write(new_content)

for root, dirs, files in os.walk("src/engine"):
    for file in files:
        if file.endswith(".hpp") or file.endswith(".cpp"):
            process_file(os.path.join(root, file))
