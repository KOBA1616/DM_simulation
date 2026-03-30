import re

with open("dm_toolkit/gui/editor/formatters/buffer_formatters.py", "r") as f:
    content = f.read()

import_stmt = "from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter\n"
if "from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter" not in content:
    content = "from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter\n" + content

def replace_buffer_qty(match):
    return """        if has_filter:
            civ_part = ''
            if civs:
                civ_part = '/'.join((CardTextResources.get_civilization_text(c) for c in civs)) + 'の'
            if races:
                type_part = '/'.join(races)
            elif 'ELEMENT' in types:
                type_part = 'エレメント'
            elif 'SPELL' in types and 'CREATURE' not in types:
                type_part = '呪文'
            elif 'CREATURE' in types:
                type_part = 'クリーチャー'
            elif types:
                type_part = '/'.join((tr(t) for t in types if t))
            else:
                type_part = 'カード'

            qty = QuantityFormatter.format_quantity(val1, "枚", up_to=False, is_all=(val1==0))
            return f'見た{civ_part}{type_part}{qty}を選び、{to_zone}に置く。'

        if val1 > 0:
            qty = QuantityFormatter.format_quantity(val1, "枚", up_to=False, is_all=False)
            return f'{qty}を{to_zone}に置く。'
        return f'選んだカードをすべて{to_zone}に置く。'"""

content = re.sub(r"        if has_filter:.*?return f'選んだカードをすべて\{to_zone\}に置く。'", replace_buffer_qty, content, flags=re.DOTALL)

with open("dm_toolkit/gui/editor/formatters/buffer_formatters.py", "w") as f:
    f.write(content)
