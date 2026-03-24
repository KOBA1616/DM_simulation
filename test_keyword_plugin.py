from dm_toolkit.gui.editor.text_generator import CardTextGenerator

card_data_rc = {
    "name": "Test Card",
    "cost": 5,
    "civilizations": ["FIRE"],
    "type": "CREATURE",
    "keywords": {
        "revolution_change": True,
        "power_attacker": True
    },
    "power_attacker_bonus": 1000,
    "effects": [
        {
            "commands": [
                {
                    "type": "REVOLUTION_CHANGE",
                    "target_filter": {
                        "civilizations": ["FIRE"],
                        "min_cost": 3
                    }
                }
            ]
        }
    ]
}

text_rc = CardTextGenerator.generate_text(card_data_rc)
print("--- RC TEXT ---")
print(text_rc)

card_data_fb = {
    "name": "Friend Burst Card",
    "cost": 4,
    "civilizations": ["WATER"],
    "type": "CREATURE",
    "keywords": {
        "friend_burst": True,
    },
    "friend_burst_condition": {
        "races": ["Magic Command"]
    }
}

text_fb = CardTextGenerator.generate_text(card_data_fb)
print("--- FB TEXT ---")
print(text_fb)
