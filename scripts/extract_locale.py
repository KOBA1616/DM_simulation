import ast
import json
import sys
import os

def extract():
    src_path = "dm_toolkit/gui/localization.py"
    if not os.path.exists(src_path):
        print(f"File not found: {src_path}")
        sys.exit(1)

    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    start_marker = "TRANSLATIONS: Dict[Any, str] = {"

    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Start marker not found")
        sys.exit(1)

    start_brace = content.find("{", start_idx)

    brace_count = 0
    end_idx = -1
    for i in range(start_brace, len(content)):
        char = content[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break

    if end_idx == -1:
        print("End marker not found")
        sys.exit(1)

    dict_str = content[start_brace:end_idx]

    lines = dict_str.split('\n')
    clean_lines = []
    for line in lines:
        if '#' in line:
            line = line.split('#')[0]
        clean_lines.append(line)
    clean_dict_str = '\n'.join(clean_lines)

    try:
        data = ast.literal_eval(clean_dict_str)
    except Exception as e:
        print(f"Failed to eval: {e}")
        sys.exit(1)

    with open("data/locale/ja.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("Success")

if __name__ == "__main__":
    extract()
