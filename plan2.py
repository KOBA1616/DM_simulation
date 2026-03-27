import re

# Update modifier_formatters.py
with open("dm_toolkit/gui/editor/formatters/modifier_formatters.py", "r") as f:
    mf_content = f.read()

mf_content = mf_content.replace(
    "from dm_toolkit.gui.editor.text_resources import CardTextResources",
    "from dm_toolkit.gui.editor.text_resources import CardTextResources"
)

old_duration_block = """        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = ""
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + "、"
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + "、"
"""
new_duration_block = """        duration_key = modifier.get('duration') or modifier.get('input_value_key', '')
        duration_text = CardTextResources.get_duration_text_with_comma(duration_key)
"""
mf_content = mf_content.replace(old_duration_block, new_duration_block)

with open("dm_toolkit/gui/editor/formatters/modifier_formatters.py", "w") as f:
    f.write(mf_content)

# Update special_effect_formatters.py
with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "r") as f:
    se_content = f.read()

old_duration_block2 = """        duration_key = command.get('duration') or command.get('input_link', '') or command.get('input_value_key', '')

        is_target_linked = bool(linked_text) and (not input_usage or input_usage == 'TARGET')

        duration_text = ''
        if duration_key:
            trans = CardTextResources.get_duration_text(duration_key)
            if trans and trans != duration_key:
                duration_text = trans + '、'
            elif duration_key in CardTextResources.DURATION_TRANSLATION:
                duration_text = CardTextResources.DURATION_TRANSLATION[duration_key] + '、'"""
new_duration_block2 = """        duration_key = command.get('duration') or command.get('input_link', '') or command.get('input_value_key', '')

        is_target_linked = bool(linked_text) and (not input_usage or input_usage == 'TARGET')

        duration_text = CardTextResources.get_duration_text_with_comma(duration_key)"""
se_content = se_content.replace(old_duration_block2, new_duration_block2)

with open("dm_toolkit/gui/editor/formatters/special_effect_formatters.py", "w") as f:
    f.write(se_content)

# Update text_resources.py
with open("dm_toolkit/gui/editor/text_resources.py", "r") as f:
    tr_content = f.read()

new_tr_func = """
    @classmethod
    def get_duration_text_with_comma(cls, duration_key: str) -> str:
        \"\"\"
        Get Japanese text for duration and append a comma if it's a known duration.
        \"\"\"
        if not duration_key:
            return ""
        trans = cls.get_duration_text(duration_key)
        if trans and trans != duration_key:
            return trans + "、"
        elif duration_key in cls.DURATION_TRANSLATION:
            return cls.DURATION_TRANSLATION[duration_key] + "、"
        return ""
"""
if "get_duration_text_with_comma" not in tr_content:
    tr_content = tr_content.replace("    @classmethod\n    def get_stat_key_label", new_tr_func + "\n    @classmethod\n    def get_stat_key_label")
    with open("dm_toolkit/gui/editor/text_resources.py", "w") as f:
        f.write(tr_content)

print("modifier_formatters.py, special_effect_formatters.py, and text_resources.py updated.")
