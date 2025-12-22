import unittest
import sys
import os

# Add bin/ to path to find dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
import dm_ai_module

class TestSelectOption(unittest.TestCase):
    def setUp(self):
        # Create a game state
        self.state = dm_ai_module.GameState(1)
        self.state.turn_number = 1
        self.state.active_player_id = 0

        # Setup dummy players
        dm_ai_module.PhaseManager.start_game(self.state, {})

        # Add a card to hand
        self.state.add_card_to_hand(0, 100, 1) # Card 100, Instance 1

        # Register dummy card data
        self.card_data = dm_ai_module.CardData(100, "Choice Card", 1, "FIRE", 1000, "CREATURE", [], [])

        # Define Option 1: Draw 1 card
        opt1_action = dm_ai_module.ActionDef()
        opt1_action.type = dm_ai_module.EffectActionType.DRAW_CARD
        opt1_action.value1 = 1
        opt1_action.target_player = "SELF"

        # Define Option 2: Add 1 to Mana
        opt2_action = dm_ai_module.ActionDef()
        opt2_action.type = dm_ai_module.EffectActionType.ADD_MANA
        opt2_action.value1 = 1
        opt2_action.target_player = "SELF"

        # Define Choice Action
        choice_action = dm_ai_module.ActionDef()
        choice_action.type = dm_ai_module.EffectActionType.SELECT_OPTION
        choice_action.options = [[opt1_action], [opt2_action]] # List of list of actions

        # Define Effect
        effect = dm_ai_module.EffectDef(
            dm_ai_module.TriggerType.ON_PLAY,
            dm_ai_module.ConditionDef(),
            [choice_action]
        )
        self.card_data.effects = [effect]

        # Mock DB
        self.card_db = {100: dm_ai_module.CardDefinition(
            100, "Choice Card", "FIRE", [], 1, 1000, dm_ai_module.CardKeywords(), [effect]
        )}
        self.card_db[100].type = dm_ai_module.CardType.CREATURE

    def test_select_option_flow(self):
        # 1. Play the card
        # Resolve Trigger ON_PLAY
        dm_ai_module.GenericCardSystem.resolve_trigger(self.state, dm_ai_module.TriggerType.ON_PLAY, 1, self.card_db)

        # Should have a pending effect (Trigger Ability)
        self.assertEqual(len(self.state.pending_effects), 1)
        self.assertEqual(self.state.pending_effects[0].type, dm_ai_module.EffectType.TRIGGER_ABILITY)

        # Resolve the Trigger Ability -> executes SELECT_OPTION action handler
        # This is done via RESOLVE_EFFECT action
        action_gen = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        # Find RESOLVE_EFFECT action
        resolve_actions = [a for a in action_gen if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]
        self.assertTrue(len(resolve_actions) > 0)

        # Execute RESOLVE_EFFECT
        dm_ai_module.EffectResolver.resolve_action(self.state, resolve_actions[0], self.card_db)

        # Now, SELECT_OPTION handler should have pushed a new pending effect of type SELECT_OPTION
        # The previous TRIGGER_ABILITY pending effect is removed.
        # Check pending effects
        # Actually, resolve_action for TRIGGER_ABILITY executes GenericCardSystem::resolve_effect -> resolve_action -> SelectOptionHandler::resolve
        # SelectOptionHandler pushes PendingEffect(SELECT_OPTION)

        self.assertEqual(len(self.state.pending_effects), 1)
        self.assertEqual(self.state.pending_effects[0].type, dm_ai_module.EffectType.SELECT_OPTION)

        # 2. Generate Actions for SELECT_OPTION
        # PendingStrategy should generate SELECT_OPTION actions for each option
        action_gen = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        select_option_actions = [a for a in action_gen if a.type == dm_ai_module.ActionType.SELECT_OPTION]

        self.assertEqual(len(select_option_actions), 2)
        self.assertEqual(select_option_actions[0].target_slot_index, 0) # Option 0
        self.assertEqual(select_option_actions[1].target_slot_index, 1) # Option 1

        # 3. Choose Option 1 (Draw 1)
        # Pre-check: hand size is 0 (we played the card, though in test setup it's still in hand unless moved, but let's assume Deck has cards)
        # Add cards to deck to draw
        self.state.add_card_to_deck(0, 200, 2)
        self.state.add_card_to_deck(0, 201, 3)
        initial_hand_size = len(self.state.players[0].hand) # 1 (the played card)

        dm_ai_module.EffectResolver.resolve_action(self.state, select_option_actions[0], self.card_db)

        # Now PendingEffect(SELECT_OPTION) is removed.
        # But wait, resolving SELECT_OPTION executes the option actions via GenericCardSystem::resolve_effect_with_context (temp effect).
        # This executes DRAW_CARD immediately (atomic) because it doesn't wait (no targets/pending).

        # Check if draw happened
        # DRAW_CARD handler executes directly.
        final_hand_size = len(self.state.players[0].hand)
        self.assertEqual(final_hand_size, initial_hand_size + 1)
        self.assertEqual(len(self.state.pending_effects), 0)

    def test_select_option_2(self):
        # 1. Play the card
        dm_ai_module.GenericCardSystem.resolve_trigger(self.state, dm_ai_module.TriggerType.ON_PLAY, 1, self.card_db)

        # Resolve Trigger
        action_gen = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        resolve_actions = [a for a in action_gen if a.type == dm_ai_module.ActionType.RESOLVE_EFFECT]
        dm_ai_module.EffectResolver.resolve_action(self.state, resolve_actions[0], self.card_db)

        # 2. Choose Option 2 (Add Mana)
        action_gen = dm_ai_module.ActionGenerator.generate_legal_actions(self.state, self.card_db)
        select_option_actions = [a for a in action_gen if a.type == dm_ai_module.ActionType.SELECT_OPTION]

        # Add card to hand to move to mana
        self.state.add_card_to_hand(0, 300, 4)
        initial_mana_size = len(self.state.players[0].mana_zone)

        # Wait, ADD_MANA action (EffectActionType::ADD_MANA) usually prompts for cards if value1 > 0.
        # It calls ManaChargeHandler::resolve -> GenericCardSystem::select_targets (TargetScope::TARGET_SELECT?)
        # Or does it auto-charge from top of deck?
        # ADD_MANA usually means "Charge from top of deck" if no source specified?
        # Let's check ManaHandler.
        # If it charges from top of deck (Accel etc), it's automatic.
        # My option definition: type=ADD_MANA, value1=1.
        # Usually this means "Put top card of deck into mana".

        self.state.add_card_to_deck(0, 400, 5)

        dm_ai_module.EffectResolver.resolve_action(self.state, select_option_actions[1], self.card_db)

        final_mana_size = len(self.state.players[0].mana_zone)
        self.assertEqual(final_mana_size, initial_mana_size + 1)

if __name__ == '__main__':
    unittest.main()
