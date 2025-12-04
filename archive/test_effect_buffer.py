
import unittest
import dm_ai_module
from dm_ai_module import GameState, CardDefinition, CardData, ActionType, EffectType, Phase, CardKeywords, EffectActionType, TargetScope, FilterDef, ActionDef, EffectDef, TriggerType, ConditionDef, CardType, Civilization

class TestEffectBuffer(unittest.TestCase):
    def setUp(self):
        self.state = GameState(42)
        # Setup dummy deck
        self.card_db = {}

        # Helper to register card
        def register(id, name, type_str, cost, civ_str, power, races, effects=[]):
            # 1. Register to CardRegistry (for GenericCardSystem)
            # CardData constructor: id, name, cost, civ, power, type, races, effects
            cdata = CardData(id, name, cost, civ_str, power, type_str, races, effects)
            dm_ai_module.register_card_data(cdata)

            # 2. Add to card_db (for ActionGenerator)
            # CardDefinition constructor: id, name, civ, races, cost, power, keywords, effects
            # We map strings to enums/constants for CardDefinition if needed, but bindings might handle strings?
            # The binding for CardDefinition init takes strings for civ and races.
            keywords = CardKeywords()
            # We pass empty effects to CardDefinition as it ignores them anyway
            cdef = CardDefinition(id, name, civ_str, races, cost, power, keywords, [])
            # Fixup type since CardDefinition constructor doesn't take type enum directly?
            # Wait, binding init doesn't take type. We set it manually.
            if type_str == "CREATURE":
                cdef.type = CardType.CREATURE
            elif type_str == "SPELL":
                cdef.type = CardType.SPELL

            # Fixup civilization enum
            if civ_str == "FIRE": cdef.civilization = Civilization.FIRE

            self.card_db[id] = cdef

        # Register normal cards
        for i in range(1, 10):
            register(i, f"Card_{i}", "CREATURE", 1, "FIRE", 1000, ["Human"])

        # Create Buffer Searcher Effect
        effect = EffectDef()
        effect.trigger = TriggerType.ON_PLAY

        # Action 1: Look to Buffer
        a1 = ActionDef()
        a1.type = EffectActionType.LOOK_TO_BUFFER
        a1.value1 = 3
        a1.source_zone = "DECK"

        # Action 2: Select from Buffer
        a2 = ActionDef()
        a2.type = EffectActionType.SELECT_FROM_BUFFER
        a2.scope = TargetScope.TARGET_SELECT
        a2.filter = FilterDef()
        a2.filter.zones = ["EFFECT_BUFFER"]
        a2.filter.count = 1
        a2.filter.types = ["CREATURE"]

        # Action 3: Play from Buffer
        a3 = ActionDef()
        a3.type = EffectActionType.PLAY_FROM_BUFFER
        a3.scope = TargetScope.TARGET_SELECT # Consumes selection

        # Action 4: Move Buffer to Zone (Bottom Deck)
        a4 = ActionDef()
        a4.type = EffectActionType.MOVE_BUFFER_TO_ZONE
        a4.destination_zone = "DECK_BOTTOM"

        effect.actions = [a1, a2, a3, a4]

        # Register Buffer Card
        register(100, "Buffer Searcher", "SPELL", 1, "FIRE", 0, [], [effect])

        dm_ai_module.initialize_card_stats(self.state, self.card_db, 40)

    def test_buffer_flow(self):
        # 1. Setup State
        # Add Buffer Searcher to hand
        self.state.add_card_to_hand(0, 100, 0) # instance_id 0

        # Add 3 creatures to deck (top of deck)
        # Deck: [Card_1, Card_2, Card_3] (top is back)
        self.state.add_card_to_deck(0, 1, 1) # Card 1
        self.state.add_card_to_deck(0, 2, 2) # Card 2
        self.state.add_card_to_deck(0, 3, 3) # Card 3

        # 2. Play Card
        self.state.current_phase = Phase.MAIN
        self.state.active_player_id = 0
        # Need at least 1 mana to play cost 1 card
        self.state.add_card_to_mana(0, 1, 999) # Add Mana

        # Generate actions
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        play_action = next((a for a in actions if a.type == ActionType.PLAY_CARD and a.card_id == 100), None)
        self.assertIsNotNone(play_action, "Should be able to play Buffer Searcher")

        # Execute Play
        dm_ai_module.EffectResolver.resolve_action(self.state, play_action, self.card_db)

        # Check pending effects (LOOK_TO_BUFFER executed, now SELECT_FROM_BUFFER is pending)
        # GameState doesn't expose pending_effects directly, use get_pending_effects_verbose
        pe_info = dm_ai_module.get_pending_effects_verbose(self.state)
        if len(pe_info) != 1:
            print("Pending Effects Info:", pe_info)
        self.assertEqual(len(pe_info), 1, "Should have pending effect for selection")
        # tuple: (type, src, ctrl, needed, current_sel_count, has_def)
        # We assume resolve_type is implicit or we can't check it directly via this helper.
        # But we can assume it works if we can generate SELECT_TARGET actions next.

        # Generate actions (SELECT_TARGET from BUFFER)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        select_actions = [a for a in actions if a.type == ActionType.SELECT_TARGET]

        self.assertEqual(len(select_actions), 3, f"Should have 3 targets in buffer. Found {len(select_actions)}")

        # Pick one (Card 2)
        chosen_action = select_actions[1] # Index 1 should be Card 2 if added in order 1,2,3?
        # Actually verify the card ID if possible, but instance_id logic is simple in test setup
        # We added 1, 2, 3 with instance ids 1, 2, 3.
        # select_actions target_instance_id should be one of 1, 2, 3.

        # Resolve Selection
        dm_ai_module.EffectResolver.resolve_action(self.state, chosen_action, self.card_db)

        # Generate actions (RESOLVE_EFFECT)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        resolve_eff = next((a for a in actions if a.type == ActionType.RESOLVE_EFFECT), None)
        self.assertIsNotNone(resolve_eff)

        # Execute Resolve (PLAY_FROM_BUFFER, MOVE_BUFFER_TO_ZONE)
        dm_ai_module.EffectResolver.resolve_action(self.state, resolve_eff, self.card_db)

        # Verify Results
        p = self.state.players[0]
        self.assertEqual(len(p.battle_zone), 1, "One creature should be in battle zone")
        self.assertEqual(len(p.deck), 2, "Two creatures should be returned to deck")
        pe_info = dm_ai_module.get_pending_effects_verbose(self.state)
        self.assertEqual(len(pe_info), 0)

if __name__ == '__main__':
    unittest.main()
