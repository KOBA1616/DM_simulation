import json, os
from pathlib import Path
from dm_toolkit.gui.editor.action_converter import ActionConverter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUT = Path(__file__).resolve().parent / 'legacy_conversion_report.json'

report = []

def convert_actions_in_obj(obj, context):
    if isinstance(obj, dict):
        # actions array
        if 'actions' in obj and isinstance(obj['actions'], list):
            for idx, act in enumerate(obj['actions']):
                if isinstance(act, dict):
                    cmd = ActionConverter.convert(act)
                    if cmd.get('legacy_warning'):
                        note = 'legacy_warning'
                    else:
                        note = ''
                    report.append({
                        'context': context,
                        'location': 'actions',
                        'index': idx,
                        'original': act,
                        'proposed_command': cmd,
                        'note': note
                    })
        # static_abilities
        if 'static_abilities' in obj and isinstance(obj['static_abilities'], list):
            for idx, act in enumerate(obj['static_abilities']):
                if isinstance(act, dict):
                    cmd = ActionConverter.convert(act)
                    report.append({
                        'context': context,
                        'location': 'static_abilities',
                        'index': idx,
                        'original': act,
                        'proposed_command': cmd,
                        'note': cmd.get('legacy_warning') and 'legacy_warning' or ''
                    })
        # triggers
        if 'triggers' in obj and isinstance(obj['triggers'], list):
            for tidx, trig in enumerate(obj['triggers']):
                if isinstance(trig, dict) and 'actions' in trig:
                    for aidx, act in enumerate(trig['actions']):
                        if isinstance(act, dict):
                            cmd = ActionConverter.convert(act)
                            report.append({
                                'context': context,
                                'location': f'triggers[{tidx}].actions',
                                'trigger_index': tidx,
                                'index': aidx,
                                'original': act,
                                'proposed_command': cmd,
                                'note': cmd.get('legacy_warning') and 'legacy_warning' or ''
                            })
        # recurse
        for k,v in obj.items():
            convert_actions_in_obj(v, context + (k,))
    elif isinstance(obj, list):
        for i,e in enumerate(obj):
            convert_actions_in_obj(e, context + (i,))


def main():
    for root,dirs,files in os.walk(DATA_DIR):
        for f in files:
            if not f.lower().endswith('.json'): continue
            fp = Path(root)/f
            try:
                data = json.loads(fp.read_text(encoding='utf-8'))
            except Exception:
                continue
            convert_actions_in_obj(data, (str(fp),))
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote {len(report)} proposed conversions to {OUT}")

if __name__=='__main__':
    main()
