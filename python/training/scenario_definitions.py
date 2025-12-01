
import dm_ai_module

SCENARIOS = {
    "infinite_loop_setup": {
        "description": "A setup where an infinite loop is possible if played correctly.",
        "config": {
            "my_mana": 6,
            "my_hand_cards": [1, 1], # Assuming ID 1 is some relevant card
            "my_battle_zone": [2],   # Assuming ID 2 is some creature
            "my_mana_zone": [3, 3, 3, 3, 3, 3],
            "my_grave_yard": [],
            "my_shields": [1, 1, 1, 1, 1],
            "enemy_shield_count": 5,
            "enemy_battle_zone": [],
            "enemy_can_use_trigger": False,
            "loop_proof_mode": True
        }
    },
    "lethal_puzzle_easy": {
        "description": "Win in one turn.",
        "config": {
            "my_mana": 3,
            "my_hand_cards": [2], # Rusher?
            "my_battle_zone": [],
            "my_mana_zone": [3, 3, 3],
            "my_shields": [],
            "enemy_shield_count": 0,
            "enemy_battle_zone": [],
            "enemy_can_use_trigger": False,
            "loop_proof_mode": False
        }
    }
}

def get_scenario_config(name):
    if name not in SCENARIOS:
        raise ValueError(f"Scenario {name} not found.")

    data = SCENARIOS[name]["config"]
    config = dm_ai_module.ScenarioConfig()

    if "my_mana" in data: config.my_mana = data["my_mana"]
    if "my_hand_cards" in data: config.my_hand_cards = data["my_hand_cards"]
    if "my_battle_zone" in data: config.my_battle_zone = data["my_battle_zone"]
    if "my_mana_zone" in data: config.my_mana_zone = data["my_mana_zone"]
    if "my_grave_yard" in data: config.my_grave_yard = data["my_grave_yard"]
    if "my_shields" in data: config.my_shields = data["my_shields"]
    if "enemy_shield_count" in data: config.enemy_shield_count = data["enemy_shield_count"]
    if "enemy_battle_zone" in data: config.enemy_battle_zone = data["enemy_battle_zone"]
    if "enemy_can_use_trigger" in data: config.enemy_can_use_trigger = data["enemy_can_use_trigger"]
    if "loop_proof_mode" in data: config.loop_proof_mode = data["loop_proof_mode"]

    return config
