
import dm_ai_module
import json
import os

SCENARIOS = {}

def load_scenarios():
    global SCENARIOS
    # Try to find data/scenarios.json
    paths = [
        "data/scenarios.json",
        "../data/scenarios.json",
        "../../data/scenarios.json"
    ]
    json_path = None
    for p in paths:
        if os.path.exists(p):
            json_path = p
            break

    if json_path:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                SCENARIOS = { item["name"]: item for item in data_list }
        except Exception as e:
            print(f"Error loading scenarios: {e}")
            SCENARIOS = {}
    else:
        print("Warning: data/scenarios.json not found.")

# Load on import
load_scenarios()

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
