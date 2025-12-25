#!/usr/bin/env python3
import re
import glob
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_cpp_enums():
    enums = {}
    pattern = re.compile(r'enum\s+class\s+(\w+)[^\{]*\{([^}]*)\}', re.MULTILINE)
    for path in glob.glob(str(ROOT / 'src' / '**' / '*.hpp'), recursive=True):
        text = Path(path).read_text(encoding='utf-8')
        for m in pattern.finditer(text):
            name = m.group(1)
            body = m.group(2)
            # split by comma, remove comments and inline initializers
            members = []
            for part in body.split(','):
                part = re.sub(r'//.*', '', part)
                part = re.sub(r'/\*.*?\*/', '', part, flags=re.S)
                part = part.strip()
                if not part:
                    continue
                # take first token (identifier)
                token = part.split()[0]
                # remove trailing = or values
                token = token.split('=')[0].strip()
                token = token.strip(',;')
                if token:
                    members.append(token)
            if name in enums:
                enums[name].extend(members)
            else:
                enums[name] = members
    # dedupe
    for k in list(enums.keys()):
        enums[k] = list(dict.fromkeys(enums[k]))
    return enums

def parse_python_consts():
    p = ROOT / 'dm_toolkit' / 'consts.py'
    source = p.read_text(encoding='utf-8')
    tree = ast.parse(source)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        items = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                items.append(str(elt.value))
                            elif isinstance(elt, ast.Str):
                                items.append(elt.s)
                        consts[name] = items
    return consts

def compare(enums, consts):
    mapping = {
        'Civilization': 'CIVILIZATIONS',
        'CardType': 'CARD_TYPES',
        'Zone': 'ZONES',
        'CommandType': 'COMMAND_TYPES',
        'EffectPrimitive': 'ACTION_TYPES'
    }
    report = []
    for cpp, py in mapping.items():
        cpp_members = enums.get(cpp, [])
        py_members = consts.get(py, [])
        report.append(f"==== {cpp} <-> {py} ====\n")
        report.append(f"C++ members ({len(cpp_members)}): {cpp_members}\n")
        report.append(f"Python consts ({len(py_members)}): {py_members}\n")
        # normalize simple differences: remove _ZONE suffix/prefix
        cpp_norm = [m for m in cpp_members]
        py_norm = [m for m in py_members]
        # For Zone, create normalized versions for fuzzy match
        if cpp == 'Zone':
            cpp_norm = [m for m in cpp_members]
            py_norm = [m.replace('_ZONE','').replace('ZONE','').replace('SHIELD','SHIELD').replace('BATTLE','BATTLE') for m in py_members]
        only_cpp = [m for m in cpp_members if m not in py_members and m not in py_norm]
        only_py = [m for m in py_members if m not in cpp_members and m not in cpp_norm]
        report.append(f"Only in C++: {only_cpp}\n")
        report.append(f"Only in Python: {only_py}\n\n")
    return '\n'.join(report)

def main():
    enums = parse_cpp_enums()
    consts = parse_python_consts()
    report = compare(enums, consts)
    out = Path('build')
    out.mkdir(exist_ok=True)
    out_file = out / 'consts_compare_report.txt'
    out_file.write_text(report, encoding='utf-8')
    print(f"Wrote report to {out_file}")

if __name__ == '__main__':
    main()
