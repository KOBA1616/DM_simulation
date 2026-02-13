import unittest
import sys
import os

# Ensure we import the local dm_ai_module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module

class TestActionGenerator(unittest.TestCase):
    def setUp(self):
        self.game = dm_ai_module.GameInstance()
        self.state = self.game.state
        self.state.initialize()

        # Mock Card DB
        self.card_db = {
            1: {"id": 1, "name": "Test Creature", "cost": 3, "type": "CREATURE", "civilization": "FIRE"},
            2: {"id": 2, "name": "Test Spell", "cost": 2, "type": "SPELL", "civilization": "WATER"},
            3: {"id": 3, "name": "Big Creature", "cost": 99, "type": "CREATURE", "civilization": "NATURE"}
        }

    def test_generator_exists(self):
        # Require command-first API to be present on the module
        self.assertTrue(callable(getattr(dm_ai_module, 'generate_commands', None)), "dm_ai_module.generate_commands not found")

    def test_mana_phase_actions(self):
        # Phase 2 = Mana
        self.state.current_phase = 2

        # Add cards to hand
        self.state.add_card_to_hand(0, 1) # Cost 3
        self.state.add_card_to_hand(0, 2) # Cost 2

        # Active player 0
        self.state.active_player_id = 0

        # Prefer command-first generator
        from dm_toolkit import commands_v2
        generate_legal_commands = commands_v2.generate_legal_commands
        cmds = generate_legal_commands(self.state, self.card_db)

        def _tname(x):
            try:
                t = x.to_dict().get('type')
            except Exception:
                t = getattr(x, 'type', None)
            try:
                return getattr(t, 'name', str(t)).upper()
            except Exception:
                return str(t).upper()

        mana_actions = [c for c in cmds if _tname(c).endswith('MANA_CHARGE')]
        pass_actions = [c for c in cmds if _tname(c) == 'PASS']

        self.assertEqual(len(mana_actions), 2)
        self.assertEqual(len(pass_actions), 1)

    def test_main_phase_actions(self):
        # Phase 3 = Main
        self.state.current_phase = 3
        self.state.active_player_id = 0

        # Add cards to hand
        self.state.add_card_to_hand(0, 1) # Cost 3
        self.state.add_card_to_hand(0, 3) # Cost 99

        # Setup Mana: 3 untaped cards
        self.state.add_card_to_mana(0, 1)
        self.state.add_card_to_mana(0, 1)
        self.state.add_card_to_mana(0, 1)
        # Ensure they are untapped (stub defaults to untapped usually, but let's be sure)
        for m in self.state.players[0].mana_zone:
            m.is_tapped = False

        from dm_toolkit import commands_v2
        generate_legal_commands = commands_v2.generate_legal_commands
        cmds = generate_legal_commands(self.state, self.card_db)

        def _t(x):
            try:
                d = x.to_dict()
                return getattr(d.get('type'), 'name', d.get('type'))
            except Exception:
                try:
                    return getattr(x, 'type', None)
                except Exception:
                    return None

        play_actions = [c for c in cmds if str(_t(c)).upper().endswith('PLAY_CARD') or str(_t(c)).upper().endswith('PLAY')]
        pass_actions = [c for c in cmds if str(_t(c)).upper() == 'PASS']

        self.assertEqual(len(play_actions), 1, "Should only be able to play the 3-cost card")
        # Check card_id via dict
        p0 = play_actions[0]
        try:
            self.assertEqual(p0.to_dict().get('card_id'), 1)
        except Exception:
            # Fallback: underlying action attribute
            self.assertEqual(getattr(p0, 'card_id', None), 1)
        self.assertEqual(len(pass_actions), 1)

    def test_attack_phase_actions(self):
        # Phase 4 = Attack
        self.state.current_phase = 4
        self.state.active_player_id = 0

        # Add creature to battle zone
        c = self.state.add_test_card_to_battle(0, 1, 100, tapped=False, sick=False)
        # Add sick creature
        c2 = self.state.add_test_card_to_battle(0, 1, 101, tapped=False, sick=True)
        # Add tapped creature
        c3 = self.state.add_test_card_to_battle(0, 1, 102, tapped=True, sick=False)

        from dm_toolkit import commands_v2
        generate_legal_commands = commands_v2.generate_legal_commands
        cmds = generate_legal_commands(self.state, self.card_db)

        attack_actions = []
        for c in cmds:
            try:
                if str(c.to_dict().get('type')).upper().endswith('ATTACK_PLAYER'):
                    attack_actions.append(c)
            except Exception:
                try:
                    if getattr(c, 'type', None) == dm_ai_module.ActionType.ATTACK_PLAYER:
                        attack_actions.append(c)
                except Exception:
                    pass

        self.assertEqual(len(attack_actions), 1)
        try:
            self.assertEqual(attack_actions[0].to_dict().get('source_instance_id'), 100)
        except Exception:
            self.assertEqual(getattr(attack_actions[0], 'source_instance_id', None), 100)

    def test_pending_effects(self):
        # Any phase
        self.state.current_phase = 3

        # Add pending effect
        self.state.pending_effects.append("DUMMY_EFFECT")

        from dm_toolkit import commands_v2
        generate_legal_commands = commands_v2.generate_legal_commands
        cmds = generate_legal_commands(self.state, self.card_db)

        self.assertEqual(len(cmds), 1)
        try:
            self.assertEqual(cmds[0].to_dict().get('type'), dm_ai_module.ActionType.RESOLVE_EFFECT)
        except Exception:
            # Fallback: underlying action attribute
            self.assertEqual(getattr(cmds[0], 'type', None), dm_ai_module.ActionType.RESOLVE_EFFECT)

if __name__ == '__main__':
    unittest.main()
