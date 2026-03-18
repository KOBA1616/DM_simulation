import re
from pathlib import Path
p = Path('logs/trace_combined.txt')
if not p.exists():
    print('trace missing')
    exit(1)
text = p.read_text(encoding='utf-8', errors='ignore')
lines = text.splitlines()
# find indexes where var_SELECTED_TARGETS appears
keywords = ['var_SELECTED_TARGETS', '$var_SELECTED_TARGETS', 'SELECTED_TARGETS']
indexes = []
for i,l in enumerate(lines):
    for kw in keywords:
        if kw in l:
            indexes.append(i)
            break
# also find PIPELINE_MOVE and [Transition] MOVED
move_idxs = [i for i,l in enumerate(lines) if 'PIPELINE_MOVE' in l or '[Transition] MOVED' in l or 'PIPELINE_EXECUTE' in l]
# collect contexts around each selected index
out = []
for idx in sorted(set(indexes)):
    start = max(0, idx-50)
    end = min(len(lines), idx+50)
    out.append('\n'.join(lines[start:end]))
# Also capture moves after the last select
if indexes:
    last = max(indexes)
    # capture following 200 lines
    start = last
    end = min(len(lines), last+400)
    out.append('\n'.join(lines[start:end]))
# fallback: if no explicit var found, search for PRE_EXEC MOVE with "count":"$var_SELECTED_TARGETS"
if not out:
    pattern = re.compile(r'count"\s*:\s*"\$var_SELECTED_TARGETS"')
    for i,l in enumerate(lines):
        if pattern.search(l):
            start = max(0, i-50)
            end = min(len(lines), i+200)
            out.append('\n'.join(lines[start:end]))
            break
# write result
op = Path('logs/trace_analysis_selected_targets.txt')
op.write_text('\n\n---- BLOCK ----\n\n'.join(out), encoding='utf-8')
print('WROTE', op)
print('Found', len(indexes), 'occurrences of keywords, captured', len(out), 'blocks')
