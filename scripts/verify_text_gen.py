from dm_toolkit.gui.editor.text_generator import CardTextGenerator


data = {
    "name": "TestCard",
    "cost": 5,
    "cost_reductions": [
        {
            "type": "PASSIVE",
            "value_mode": "FIXED",
            "value": 2,
            "unit_cost": {"filter": {"civilizations": ["DARKNESS"], "min_cost": 0}}
        }
    ]
}

print("--- FIXED ---")
print(CardTextGenerator.generate_body_text(data))

data2 = {
    "name": "StatCard",
    "cost": 8,
    "cost_reductions": [
        {
            "type": "PASSIVE",
            "value_mode": "STAT_SCALED",
            "stat_key": "TOTAL_POWER",
            "per_value": 3,
            "increment_cost": 1,
            "max_reduction": 4,
            "unit_cost": {"filter": {"civilizations": ["DARKNESS"]}}
        }
    ]
}
print("--- STAT_SCALED ---")
print(CardTextGenerator.generate_body_text(data2))
