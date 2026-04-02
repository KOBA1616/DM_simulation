import json
from dm_toolkit.gui.editor.formatters.keyword_registry import SpecialKeywordRegistry
from dm_toolkit.gui.editor.formatters.special_keywords import DangerousDashFormatter
from dm_toolkit.gui.editor.text_resources import CardTextResources

formatter = SpecialKeywordRegistry.get_formatter("dangerous_dash")
assert formatter is not None

card_data_1 = {
    "dangerous_dash_condition": {
        "civilizations": ["FIRE"],
        "cost": 4
    }
}

card_data_2 = {
    "dangerous_dash_condition": {
        "raw_text": "火のカードを1枚自分の手札から捨てる"
    }
}

print(formatter.format("dangerous_dash", card_data_1))
print(formatter.format("dangerous_dash", card_data_2))
