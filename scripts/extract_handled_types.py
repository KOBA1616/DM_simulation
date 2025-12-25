import re
p = re.compile(r"act_type\s*(?:==|in)\s*(?:\(|)?\s*('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")")
text = open('dm_toolkit/gui/editor/action_converter.py', encoding='utf-8').read()
found = set()
# simple matches for == 'FOO' and in ('A','B') etc.
for m in re.finditer(r"act_type\s*==\s*(['\"])(.*?)\1", text):
    found.add(m.group(2))
for m in re.finditer(r"act_type\s*in\s*\(([^\)]*)\)", text):
    inner = m.group(1)
    for s in re.findall(r"['\"](.*?)['\"]", inner):
        found.add(s)
# also check lines with elif act_type in ["A","B"]
for m in re.finditer(r"act_type\s*in\s*\[([^\]]*)\]", text):
    inner = m.group(1)
    for s in re.findall(r"['\"](.*?)['\"]", inner):
        found.add(s)
# also search for lines like if act_type == "MOVE_CARD":
for m in re.finditer(r"if\s+act_type\s*==\s*(['\"])(.*?)\1", text):
    found.add(m.group(2))
# print sorted
for t in sorted(found):
    print(t)
