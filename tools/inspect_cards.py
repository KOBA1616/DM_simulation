"""カード効果とテキストの整合性を調査するスクリプト"""
import json
import sys

def show_cmds(cmds, indent=6):
    for cmd in cmds:
        sp = ' ' * indent
        fields = {k: v for k, v in cmd.items() if v is not None and v != [] and v != {}}
        # 主要フィールドのみ表示
        core = {k: v for k, v in fields.items()
                if k not in ('sub_commands', 'if_true', 'if_false', 'options', 'target_filter', 'condition_filter')}
        print(f"{sp}CMD type={cmd.get('type')} {core}")
        if cmd.get('target_filter'):
            print(f"{sp}  target_filter={cmd['target_filter']}")
        if cmd.get('condition_filter'):
            print(f"{sp}  condition_filter={cmd['condition_filter']}")
        if cmd.get('sub_commands'):
            print(f"{sp}  sub_commands:")
            show_cmds(cmd['sub_commands'], indent + 4)
        if cmd.get('if_true'):
            print(f"{sp}  if_true:")
            show_cmds(cmd['if_true'], indent + 4)
        if cmd.get('if_false'):
            print(f"{sp}  if_false:")
            show_cmds(cmd['if_false'], indent + 4)
        if cmd.get('options'):
            for opt in cmd['options']:
                if isinstance(opt, dict):
                    print(f"{sp}  option[{opt.get('label')}]:")
                    show_cmds(opt.get('commands', []), indent + 4)
                elif isinstance(opt, list):
                    # optionsがlist of listの場合
                    print(f"{sp}  option_list:")
                    show_cmds(opt, indent + 4)

with open('data/cards.json', encoding='utf-8') as f:
    cards = json.load(f)

for card in cards:
    cid = card['id']
    name = card['name']
    ctype = card.get('type', '?')
    cost = card.get('cost', '?')
    power = card.get('power', '-')
    text = card.get('text', '').replace('\n', ' | ')
    print(f"=== Card {cid}: {name} ===")
    print(f"  type={ctype}, cost={cost}, power={power}")
    print(f"  card text: {text!r}")
    effects = card.get('effects', [])
    if not effects:
        print("  [effects: NONE]")
    for eff in effects:
        timing = eff.get('timing', '')
        trigger = eff.get('trigger_event', '')
        condition = eff.get('condition', '')
        print(f"  [effect timing={timing!r} trigger={trigger!r} condition={condition!r}]")
        show_cmds(eff.get('commands', []))
    print()
