with open("tests/test_effect_and_text_integrity.py", "r") as f:
    test_content = f.read()

test_content = test_content.replace(
    'from dm_toolkit.gui.editor.text_generator import CardTextGenerator',
    'from dm_toolkit.gui.editor.text_generator import CardTextGenerator\nfrom dm_toolkit.gui.editor.formatters.filter_formatter import FilterTextFormatter'
)

with open("tests/test_effect_and_text_integrity.py", "w") as f:
    f.write(test_content)
