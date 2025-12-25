import json,os,re,sys
p=os.path.join(os.path.dirname(__file__), '..', 'data')
if not os.path.isdir(p):
    p = os.path.join(os.path.dirname(__file__), 'data')
p = os.path.abspath(p)
print('scanning', p)
types=set()
for root,dirs,files in os.walk(p):
    for f in files:
        if not f.lower().endswith('.json'): continue
        fp=os.path.join(root,f)
        try:
            with open(fp,'r',encoding='utf-8') as fh:
                data=json.load(fh)
        except Exception as e:
            try:
                txt=open(fp,'r',encoding='utf-8').read()
                for m in re.finditer(r'"type"\s*:\s*"([A-Z_]+)"', txt):
                    types.add(m.group(1))
                continue
            except Exception:
                continue
        def walk(o):
            if isinstance(o,dict):
                for k,v in o.items():
                    if k=='type' and isinstance(v,str) and re.fullmatch(r'[A-Z_]+',v):
                        types.add(v)
                    walk(v)
            elif isinstance(o,list):
                for e in o: walk(e)
        walk(data)

for t in sorted(types):
    print(t)
