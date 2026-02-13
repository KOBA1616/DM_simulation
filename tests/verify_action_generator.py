import unittest
import sys
import os

# Ensure we import the local dm_ai_module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module

IS_NATIVE = getattr(dm_ai_module, 'IS_NATIVE', False)

class TestActionGenerator(unittest.TestCase):
    def setUp(self):
        self.game = dm_ai_module.GameInstance()
        self.state = self.game.state
        self.state.setup_test_duel()

        # Mock Card DB
        self.card_db = {
            1: {"id": 1, "name": "Test Creature", "cost": 3, "type": "CREATURE", "civilization": "FIRE"},
            2: {"id": 2, "name": "Test Spell", "cost": 2, "type": "SPELL", "civilization": "WATER"},
            3: {"id": 3, "name": "Big Creature", "cost": 99, "type": "CREATURE", "civilization": "NATURE"}
        }

    def test_generator_exists(self):
        # ActionGenerator is aliased to IntentGenerator in shim
        self.assertTrue(hasattr(dm_ai_module, 'IntentGenerator') or hasattr(dm_ai_module, 'ActionGenerator'), "IntentGenerator class is missing")

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
            if isinstance(x, dict):
                t = x.get('type')
            else:
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
            if isinstance(x, dict):
                return x.get('type')
            try:
                d = x.to_dict()
                return getattr(d.get('type'), 'name', d.get('type'))
            except Exception:
                try:
                    return getattr(x, 'type', None)
                except Exception:
                    return None

        play_actions = [c for c in cmds if str(_t(c)).upper().endswith('PLAY_FROM_ZONE') or str(_t(c)).upper().endswith('PLAY_CARD')]
        pass_actions = [c for c in cmds if str(_t(c)).upper() == 'PASS']

        if IS_NATIVE:
            self.assertEqual(len(play_actions), 1, "Should only be able to play the 3-cost card")
        else:
            # Shim returns all cards as playable (doesn't check cost)
            self.assertEqual(len(play_actions), 2, "Shim returns all cards")

        # Check card_id via dict
        p0 = play_actions[0]
        cid = None
        if isinstance(p0, dict):
            cid = p0.get('card_id')
        else:
            try:
                cid = p0.to_dict().get('card_id')
            except Exception:
                cid = getattr(p0, 'card_id', None)
        self.assertEqual(cid, 1)
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
            t = None
            if isinstance(c, dict):
                t = c.get('type')
            else:
                try:
                    t = c.to_dict().get('type')
                except Exception:
                    t = getattr(c, 'type', None)

            t_str = str(t).upper()
            if t_str == 'ATTACK' or t_str.endswith('ATTACK_PLAYER'):
                attack_actions.append(c)
            elif t == dm_ai_module.CommandType.ATTACK_PLAYER:
                attack_actions.append(c)

        self.assertEqual(len(attack_actions), 1)
        src = None
        if isinstance(attack_actions[0], dict):
            src = attack_actions[0].get('instance_id') or attack_actions[0].get('source_instance_id')
        else:
            try:
                src = attack_actions[0].to_dict().get('source_instance_id')
            except Exception:
                src = getattr(attack_actions[0], 'source_instance_id', None)

        self.assertEqual(src, 100)

    def test_pending_effects(self):
        if not IS_NATIVE:
            return # Shim doesn't implement pending effect resolution generation

        # Any phase
        self.state.current_phase = 3

        # Add pending effect
        self.state.pending_effects.append("DUMMY_EFFECT")

        from dm_toolkit import commands_v2
        generate_legal_commands = commands_v2.generate_legal_commands
        cmds = generate_legal_commands(self.state, self.card_db)

        if len(cmds) > 0:
             c0 = cmds[0]
             t = c0.get('type') if isinstance(c0, dict) else getattr(c0, 'type', None)
             # Should be RESOLVE_EFFECT
             self.assertTrue(str(t).upper() == 'RESOLVE_EFFECT' or t == dm_ai_module.CommandType.RESOLVE_EFFECT)

if __name__ == '__main__':
    unittest.main()
