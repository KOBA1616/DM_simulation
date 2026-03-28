import os
import glob
import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacements for `_format_game_action_command(cmd_type, action, is_spell, look_count, add_count, target_str, unit, ...)`
    # This is quite complex to regex perfectly due to multi-line function calls.
    # We will use a generalized regex to replace `CardTextGenerator._format_game_action_command` and `CardTextGenerator._format_buffer_command`
    # with the appropriate mock context and `CardTextGenerator._format_command` if possible.

    # Actually, the simplest fix for these unit tests is to mock a context and call the formatter directly
    # However, since they test CardTextGenerator, we should pass a command dict to `CardTextGenerator._format_command(cmd, ctx)`

    # A simpler approach: Since `CardTextGenerator` relies on `TextGenerationContext`, we can just mock the context
    # in the test and call `_format_command`.
    pass

# We will just rewrite the failing tests to use `CardTextGenerator.generate_text({"effects": [{"commands": [...]}]})` or similar
# Let's inspect a few failing tests to see how they are structured.
