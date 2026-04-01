import re

with open("dm_toolkit/gui/editor/formatters/condition_formatter.py", "r") as f:
    content = f.read()

def repl(match):
    return """class CompareInputConditionFormatter(ConditionFormatterStrategy):
    @classmethod
    def format(cls, d: Dict[str, Any], ctx: TextGenerationContext = None) -> str:
        # action is no longer passed as a separate param, we use ctx to pull input references if needed.
        # However, for compatibility we extract it from `d` or `ctx` safely.
        action = d if d.get("input_link") or d.get("input_value_key") else {}
        val = d.get("value", 0)
        op = d.get("op", ">=")

        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

        input_key = action.get("input_value_key") or action.get("input_link") or ""

        input_desc = InputLinkFormatter.resolve_linked_value_text(action, context_commands=ctx.current_commands_list if ctx else None)
        if not input_desc:
            input_desc_map = {
                "spell_count": "墓地の呪文の数",
                "card_count": "カードの数",
                "creature_count": "クリーチャーの数",
                "element_count": "エレメントの数"
            }
            input_desc = input_desc_map.get(input_key, InputLinkFormatter.format_input_source_label(action) or "入力値")

        ival = int(val) if isinstance(val, (int, str)) and str(val).isdigit() else val

        # Special logic: for some ops (like >= with an incremented ival), we adapt the value first
        if op == ">=":
             val_str = f"{ival + 1}" if isinstance(ival, int) else f"{val}"
        else:
             val_str = f"{val}"

        op_text = TextUtils.format_comparison_operator(op, val_str, attribute=input_desc, particle="が")

        return f"{op_text}なら\""""

content = re.sub(r"class CompareInputConditionFormatter\(ConditionFormatterStrategy\):.*?return f\"\{input_desc\}が\{op_text\}なら\"", repl, content, flags=re.DOTALL)

with open("dm_toolkit/gui/editor/formatters/condition_formatter.py", "w") as f:
    f.write(content)
