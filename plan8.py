with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "r") as f:
    gaf_content = f.read()

gaf_content = gaf_content.replace(
"""            input_key = command.get('input_value_key') or command.get('input_link')
        if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                if usage_label:
                    base += f'（{usage_label}）'
            return base""",
"""            input_key = command.get('input_value_key') or command.get('input_link')
            if input_key:
                usage_label = InputLinkFormatter.format_input_usage_label(input_usage)
                if usage_label:
                    base += f'（{usage_label}）'
            return base""")

with open("dm_toolkit/gui/editor/formatters/game_action_formatters.py", "w") as f:
    f.write(gaf_content)

print("Fixed indentation error.")
