with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "r") as f:
    gaf_content = f.read()

gaf_content = gaf_content.replace(
"""            input_key = command.get('input_value_key') or command.get('input_link')
        if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                cnt_txt = '指定数'
                if usage_label:
                    cnt_txt = f'入力値（{usage_label}）'
                if filter_txt:
                    return f'{filter_txt}から{cnt_txt}選ぶ。'
                return f'条件に合うカードから{cnt_txt}選ぶ。'
            if filter_txt:
                return f'{filter_txt}から{sel_count}枚選ぶ。'
            return f'条件に合うカードから{sel_count}枚選ぶ。'""",
"""            input_key = command.get('input_value_key') or command.get('input_link')
            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                cnt_txt = '指定数'
                if usage_label:
                    cnt_txt = f'入力値（{usage_label}）'
                if filter_txt:
                    return f'{filter_txt}から{cnt_txt}選ぶ。'
                return f'条件に合うカードから{cnt_txt}選ぶ。'
            if filter_txt:
                return f'{filter_txt}から{sel_count}枚選ぶ。'
            return f'条件に合うカードから{sel_count}枚選ぶ。'""")

with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "w") as f:
    f.write(gaf_content)
