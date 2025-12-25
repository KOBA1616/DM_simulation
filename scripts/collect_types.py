import glob, json

types=set()
for fp in glob.glob('data/**/*.json', recursive=True):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            j=json.load(f)
    except Exception:
        continue
    def walk(x):
        if isinstance(x, dict):
            if 'type' in x and isinstance(x['type'], str):
                types.add(x['type'])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for e in x:
                walk(e)
    walk(j)
for t in sorted(types):
    print(t)
