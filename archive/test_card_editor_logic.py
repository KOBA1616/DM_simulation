import json
import os
import unittest
import sys

# Add bin/ to path to import dm_ai_module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
bin_path = os.path.join(project_root, "bin")
if bin_path not in sys.path:
    sys.path.append(bin_path)

import dm_ai_module

class ShadowCardEditor:
    """
    A shadow class that replicates the data manipulation logic of the GUI CardEditor
    without depending on PyQt/QWidgets. This allows testing the logic of card creation
    and modification.
    """
    def __init__(self):
        self.cards_data = []
        self.current_card_index = -1

        # Mock UI state
        self.id_input = 0
        self.name_input = ""
        self.civ_input = "FIRE"
        self.type_input = "CREATURE"
        self.cost_input = 0
        self.power_input = 0
        self.races_input = ""

        self.keyword_checkboxes = {
            "BLOCKER": False, "SPEED_ATTACKER": False, "SLAYER": False,
            "DOUBLE_BREAKER": False, "TRIPLE_BREAKER": False, "POWER_ATTACKER": False,
            "EVOLUTION": False, "MACH_FIGHTER": False, "G_STRIKE": False,
            "JUST_DIVER": False
        }

        self.shield_trigger_cb = False
        self.hyper_energy_cb = False

        self.rev_change_group_checked = False
        self.rev_civ_input = ""
        self.rev_race_input = ""
        self.rev_cost_spin = 5

    def create_new_card(self):
        new_id = 1
        if self.cards_data:
            new_id = max(c.get('id', 0) for c in self.cards_data) + 1

        new_card = {
            "id": new_id,
            "name": "New Card",
            "civilization": "FIRE",
            "type": "CREATURE",
            "cost": 1,
            "power": 1000,
            "races": [],
            "effects": []
        }
        self.cards_data.append(new_card)
        self.current_card_index = len(self.cards_data) - 1

    def update_current_card_data(self):
        if self.current_card_index < 0:
            return

        card = self.cards_data[self.current_card_index]

        card['id'] = self.id_input
        card['name'] = self.name_input
        card['civilization'] = self.civ_input
        card['type'] = self.type_input
        card['cost'] = self.cost_input
        card['power'] = self.power_input

        races_str = self.races_input
        card['races'] = [r.strip() for r in races_str.split(',')] if races_str.strip() else []

        # Sync Shield Trigger
        if self.shield_trigger_cb:
            if 'keywords' not in card: card['keywords'] = {}
            card['keywords']['shield_trigger'] = True
        else:
            if 'keywords' in card and 'shield_trigger' in card['keywords']:
                del card['keywords']['shield_trigger']

        # Sync Hyper Energy
        # 1. Remove existing Hyper Energy actions/effects
        cleaned_effects_for_hyper = []
        for eff in card.get('effects', []):
            new_actions = []
            for act in eff.get('actions', []):
                if not (act.get('type') == 'COST_REFERENCE' and act.get('str_val') == 'HYPER_ENERGY'):
                    new_actions.append(act)

            if new_actions or eff.get('trigger') != 'NONE':
                eff['actions'] = new_actions
                cleaned_effects_for_hyper.append(eff)

        card['effects'] = cleaned_effects_for_hyper

        # 2. Add back if checked
        if self.hyper_energy_cb:
            card['effects'].append({
                "trigger": "NONE",
                "condition": {"type": "NONE"},
                "actions": [{
                    "type": "COST_REFERENCE",
                    "str_val": "HYPER_ENERGY",
                    "value1": 0,
                    "scope": "PLAYER_SELF",
                    "filter": {}
                }]
            })

        # Sync Revolution Change
        if self.rev_change_group_checked:
            rev_cond = {}
            civs = [c.strip() for c in self.rev_civ_input.split(',') if c.strip()]
            if civs: rev_cond['civilizations'] = civs
            races = [r.strip() for r in self.rev_race_input.split(',') if r.strip()]
            if races: rev_cond['races'] = races
            min_cost = self.rev_cost_spin
            if min_cost > 0: rev_cond['min_cost'] = min_cost
            card['revolution_change_condition'] = rev_cond
        else:
            if 'revolution_change_condition' in card:
                del card['revolution_change_condition']

        # Sync Keyword Checkboxes to PASSIVE_CONST
        new_effects = []
        existing_effects = card.get('effects', [])

        known_keywords = set(self.keyword_checkboxes.keys())

        # Keep existing non-keyword effects
        for eff in existing_effects:
            if eff.get('trigger') == 'PASSIVE_CONST':
                actions_to_keep = []
                for act in eff.get('actions', []):
                    if act.get('str_val') not in known_keywords:
                        actions_to_keep.append(act)
                if actions_to_keep:
                    eff['actions'] = actions_to_keep
                    new_effects.append(eff)
            else:
                new_effects.append(eff)

        # Add checked keywords
        active_kws = []
        for kw, checked in self.keyword_checkboxes.items():
            if checked:
                active_kws.append(kw)

        if active_kws:
            actions = []
            for kw in active_kws:
                actions.append({
                    "type": "NONE",
                    "scope": "NONE",
                    "filter": {},
                    "value1": 0,
                    "value2": 0,
                    "str_val": kw
                })
            new_effects.append({
                "trigger": "PASSIVE_CONST",
                "condition": {"type": "NONE", "value": 0, "str_val": ""},
                "actions": actions
            })

        card['effects'] = new_effects


class TestCardEditorLogic(unittest.TestCase):
    def setUp(self):
        self.editor = ShadowCardEditor()
        self.temp_file = "temp_test_cards_logic.json"

    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_create_complex_card(self):
        # 1. Create New Card
        self.editor.create_new_card()

        # 2. Set Properties
        self.editor.id_input = 1001
        self.editor.name_input = "Hyper Dragon"
        self.editor.civ_input = "FIRE"
        self.editor.type_input = "CREATURE"
        self.editor.cost_input = 5
        self.editor.power_input = 6000
        self.editor.races_input = "Armored Dragon, Fire Bird"

        # Keywords
        self.editor.keyword_checkboxes["SPEED_ATTACKER"] = True
        self.editor.keyword_checkboxes["DOUBLE_BREAKER"] = True
        self.editor.keyword_checkboxes["JUST_DIVER"] = True  # New feature

        # Shield Trigger
        self.editor.shield_trigger_cb = True

        # Hyper Energy
        self.editor.hyper_energy_cb = True

        # Revolution Change
        self.editor.rev_change_group_checked = True
        self.editor.rev_civ_input = "FIRE"
        self.editor.rev_race_input = "Dragon"
        self.editor.rev_cost_spin = 5

        # Apply Updates
        self.editor.update_current_card_data()

        # 3. Add Custom Effect (simulate adding manually via effect editor logic)
        card = self.editor.cards_data[0]

        # Add a custom effect like "ON_PLAY destroy creature"
        custom_effect = {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "actions": [{
                "type": "DESTROY",
                "scope": "TARGET_SELECT",
                "value1": 1,
                "filter": {"zones": ["BATTLE_ZONE"], "owner": "PLAYER_OPPONENT"}
            }]
        }
        card['effects'].append(custom_effect)

        # Update again to ensure nothing gets wiped
        self.editor.update_current_card_data()

        # 4. Save to JSON
        with open(self.temp_file, 'w', encoding='utf-8') as f:
            json.dump(self.editor.cards_data, f, indent=2)

        # 5. Load with Engine
        db = dm_ai_module.JsonLoader.load_cards(self.temp_file)

        self.assertIn(1001, db)
        loaded_card = db[1001]

        # Verify Basic Info
        self.assertEqual(loaded_card.name, "Hyper Dragon")
        self.assertEqual(loaded_card.civilization, dm_ai_module.Civilization.FIRE)
        self.assertEqual(loaded_card.cost, 5)
        self.assertEqual(loaded_card.power, 6000)
        self.assertIn("Armored Dragon", loaded_card.races)

        # Verify Keywords (exposed via properties in binding)
        self.assertTrue(loaded_card.keywords.speed_attacker)
        self.assertTrue(loaded_card.keywords.double_breaker)
        # JUST_DIVER might not be exposed as property, but checked via effects below

        # Verify Shield Trigger
        self.assertTrue(loaded_card.keywords.shield_trigger)

        # Verify Revolution Change
        self.assertTrue(loaded_card.keywords.revolution_change)
        rev_cond = loaded_card.revolution_change_condition
        self.assertIsNotNone(rev_cond)

        # Verify Effects via JSON inspection
        with open(self.temp_file, 'r') as f:
            json_data = json.load(f)[0]

        effects = json_data['effects']

        # Check Hyper Energy
        hyper_eff = next((e for e in effects if any(a.get('str_val') == 'HYPER_ENERGY' for a in e.get('actions', []))), None)
        self.assertIsNotNone(hyper_eff, "Hyper Energy effect not found in JSON")

        # Check Keywords
        kw_eff = next((e for e in effects if e.get('trigger') == 'PASSIVE_CONST'), None)
        self.assertIsNotNone(kw_eff, "Keywords PASSIVE_CONST effect not found in JSON")
        kw_actions = [a.get('str_val') for a in kw_eff['actions']]
        self.assertIn("SPEED_ATTACKER", kw_actions)
        self.assertIn("JUST_DIVER", kw_actions)

        # Check Revolution Change JSON
        self.assertIn("revolution_change_condition", json_data)
        self.assertEqual(json_data['revolution_change_condition']['min_cost'], 5)

if __name__ == '__main__':
    unittest.main()
