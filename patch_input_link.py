import re

with open("dm_toolkit/gui/editor/formatters/input_link_formatter.py", "r") as f:
    content = f.read()

def repl(m):
    return """    @classmethod
    def resolve_linked_value_text(cls, command: Dict[str, Any], default: str = "", context_commands: Optional[List[Dict[str, Any]]] = None) -> str:
        \"\"\"Resolve the linked value text for a given command.

        Returns a string like 'その数' or 'そのコストと同じ' if the command has an input link,
        otherwise returns the default string provided. AST is used if context_commands is given.
        \"\"\"
        input_key = str(command.get("input_value_key") or command.get("input_link") or "")
        if not input_key:
            return default

        usage = str(command.get("input_usage") or command.get("input_value_usage") or "").upper()
        ctype_self = command.get("type", "")

        # Target reference shortcuts: If modifying targets dynamically
        if usage == "TARGET_SELECTION" or ctype_self in ["DESTROY", "TAP", "UNTAP", "RETURN_TO_HAND", "SEND_TO_MANA"]:
             up_to_flag = bool(command.get('up_to', False))
             if up_to_flag:
                  return "その同じ数だけまで選び"
             return "その同じ数だけ"

        if usage in ["COUNT", "AMOUNT"]:
             return "その同じ数"

        # Context-aware resolution using AST"""

content = re.sub(r"    @classmethod\n    def resolve_linked_value_text.*?# Context-aware resolution using AST", repl, content, flags=re.DOTALL)

with open("dm_toolkit/gui/editor/formatters/input_link_formatter.py", "w") as f:
    f.write(content)
