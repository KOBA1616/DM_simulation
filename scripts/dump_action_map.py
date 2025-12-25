import json
from dm_toolkit.gui.editor import text_generator

# Extract ACTION_MAP keys and any special handlers noted in the class
action_map = getattr(text_generator.CardTextGenerator, 'ACTION_MAP', {})

dump = { 'action_map_keys': list(action_map.keys()) }
print(json.dumps(dump, ensure_ascii=False, indent=2))
