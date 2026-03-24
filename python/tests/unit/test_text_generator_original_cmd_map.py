from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_shield_trigger_command_returns_short_text():
    cmd = {"type": "SHIELD_TRIGGER"}
    text = CardTextGenerator._format_command(cmd)
    assert text == "S・トリガー"
