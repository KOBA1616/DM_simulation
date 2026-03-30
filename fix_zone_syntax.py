import re

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "r") as f:
    content = f.read()

content = content.replace("from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter\\nfrom dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter\\nfrom dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter\\n", "        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter\\n")

# Just do a clean fix:
content = content.replace("""        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
from dm_toolkit.gui.editor.formatters.quantity_formatter import QuantityFormatter
from dm_toolkit.gui.editor.formatters.zone_formatter import ZoneFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)""", """        from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter
        linked_text = InputLinkFormatter.resolve_linked_value_text(command, context_commands=ctx.current_commands_list)""")

with open("dm_toolkit/gui/editor/formatters/zone_formatters.py", "w") as f:
    f.write(content)
