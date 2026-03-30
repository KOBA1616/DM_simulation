import re

with open("dm_toolkit/gui/editor/text_generator.py", "r") as f:
    content = f.read()

def replace_modifier_gen(match):
    return """        from dm_toolkit.gui.editor.formatters.modifier_formatters import ModifierFormatterRegistry
        if mtype:
            ModifierFormatterRegistry.update_metadata(mtype, modifier, ctx)

        # Delegate fully to TargetResolutionService to build "自分のクリーチャー" etc.
        from dm_toolkit.gui.editor.services.target_resolution_service import TargetResolutionService
        full_target = TargetResolutionService.format_modifier_target(filter_def, scope=scope)

        return ModifierFormatterRegistry.format(mtype, cond_text, full_target, value, modifier, ctx)"""

content = re.sub(r"        from dm_toolkit\.gui\.editor\.formatters\.modifier_formatters import ModifierFormatterRegistry.*?ModifierFormatterRegistry\.format\(mtype, cond_text, full_target, scope_prefix, value, modifier, ctx\)", replace_modifier_gen, content, flags=re.DOTALL)

with open("dm_toolkit/gui/editor/text_generator.py", "w") as f:
    f.write(content)
