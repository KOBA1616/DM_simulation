import os, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'

def find_objects(obj, path=()):
    # Yield (parent_obj, key, obj, path)
    if isinstance(obj, dict):
        yield (None, None, obj, path)
        for k,v in obj.items():
            for res in find_objects(v, path + (k,)):
                yield res
    elif isinstance(obj, list):
        for i, e in enumerate(obj):
            for res in find_objects(e, path + (i,)):
                yield res


def scan_file(fp):
    try:
        data = json.loads(open(fp,'r',encoding='utf-8').read())
    except Exception:
        txt = open(fp,'r',encoding='utf-8',errors='ignore').read()
        # fallback quick regex search
        entries = []
        for m in re.finditer(r'"type"\s*:\s*"([A-Z_]+)"', txt):
            entries.append({'file': str(fp), 'match_type': m.group(1)})
        return entries

    results = []
    # Walk looking for 'actions', 'static_abilities', 'triggers', 'reaction_abilities'
    def walk(obj, context):
        if isinstance(obj, dict):
            # if this dict looks like a card or template, capture id/name
            meta = {}
            if 'id' in obj:
                meta['id'] = obj.get('id')
            if 'name' in obj:
                meta['name'] = obj.get('name')
            # actions array
            if 'actions' in obj and isinstance(obj['actions'], list):
                for idx, act in enumerate(obj['actions']):
                    if isinstance(act, dict):
                        t = act.get('type','NONE')
                        # consider legacy when actions contain non-command types or when type==NONE+str_val
                        if t in ['DRAW_CARD','DESTROY','SEND_TO_MANA','FRIEND_BURST','REVOLUTION_CHANGE','PLAY_FROM_ZONE','GRANT_KEYWORD','NINJA_STRIKE','NONE','OPPONENT_DRAW_COUNT','QUERY','BRANCH']:
                            results.append({'file': str(fp), 'context': context, 'meta': meta, 'action_index': idx, 'action_type': t, 'action': act})
            # static_abilities may contain action-like dicts
            if 'static_abilities' in obj and isinstance(obj['static_abilities'], list):
                for idx, act in enumerate(obj['static_abilities']):
                    if isinstance(act, dict) and act.get('type'):
                        t = act.get('type','NONE')
                        if t in ['GRANT_KEYWORD','REVOLUTION_CHANGE','NONE']:
                            results.append({'file': str(fp), 'context': context+('static_abilities',), 'meta': meta, 'action_index': idx, 'action_type': t, 'action': act})
            # triggers
            if 'triggers' in obj and isinstance(obj['triggers'], list):
                for tidx, trig in enumerate(obj['triggers']):
                    if isinstance(trig, dict):
                        if 'actions' in trig and isinstance(trig['actions'], list):
                            for aidx, act in enumerate(trig['actions']):
                                if isinstance(act, dict):
                                    t = act.get('type','NONE')
                                    if t in ['DRAW_CARD','DESTROY','SEND_TO_MANA','FRIEND_BURST','REVOLUTION_CHANGE','PLAY_FROM_ZONE','GRANT_KEYWORD','NINJA_STRIKE','NONE','OPPONENT_DRAW_COUNT','QUERY','BRANCH']:
                                        results.append({'file': str(fp), 'context': context+('triggers',tidx), 'meta': meta, 'trigger_index': tidx, 'action_index': aidx, 'action_type': t, 'action': act})
        elif isinstance(obj, list):
            for i,e in enumerate(obj):
                walk(e, context + (i,))

    walk(data, ())
    return results


def main():
    out = []
    for root,dirs,files in os.walk(DATA_DIR):
        for f in files:
            if not f.lower().endswith('.json'): continue
            fp = Path(root)/f
            res = scan_file(fp)
            if res:
                out.extend(res)
    # Print summarized report
    if not out:
        print('No legacy-looking actions found.')
        return
    for e in out:
        file = e.get('file')
        meta = e.get('meta',{})
        name = meta.get('name') or meta.get('id') or Path(file).name
        atype = e.get('action_type')
        idx = e.get('action_index')
        trig = e.get('trigger_index', None)
        info = f"{file} | {name} | trigger={trig} | action_idx={idx} | type={atype}"
        print(info)
        # pretty print action keys
        try:
            import pprint
            pprint.pprint(e.get('action'))
        except Exception:
            print(e.get('action'))
        print('-'*60)

if __name__=='__main__':
    main()
