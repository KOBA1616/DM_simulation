
import unittest
from dm_ai_module import GameState, CardType, GenericCardSystem, FilterDef, CardDefinition, CardData, Civilization, CardKeywords, EffectDef, ActionDef, EffectActionType, TargetScope

class TestTamaseed(unittest.TestCase):
    def test_tamaseed_logic(self):
        # Create a mock GameState
        state = GameState(100)
        state.turn_number = 1
        state.active_player_id = 0

        # Create a Tamaseed definition
        # ID 1000
        tamaseed_def = CardDefinition()
        tamaseed_def.id = 1000
        tamaseed_def.name = "Test Tamaseed"
        tamaseed_def.type = CardType.TAMASEED
        tamaseed_def.cost = 2
        tamaseed_def.power = 0
        tamaseed_def.civilizations = [Civilization.LIGHT]

        # Manually register the card if possible or just use a local map for testing
        # GenericCardSystem::resolve_effect uses CardRegistry::get_all_definitions() usually.
        # But we can test TargetUtils logic by manually invoking GenericCardSystem helpers if they exposed TargetUtils.
        # TargetUtils is not exposed, but FilterDef is.
        # We can't directly call TargetUtils.is_valid_target from Python.

        # However, we can use GenericCardSystem.resolve_action with COUNT_CARDS to verify counting.

        # Add Tamaseed to Battle Zone
        state.add_test_card_to_battle(0, 1000, 1, False, False)

        # 1. Verify "ELEMENT" counts it
        # Filter: ELEMENT, Zone: BATTLE
        filter_elem = FilterDef()
        filter_elem.zones = ["BATTLE_ZONE"]
        filter_elem.types = ["ELEMENT"]

        action_count_elem = ActionDef()
        action_count_elem.type = EffectActionType.COUNT_CARDS
        action_count_elem.filter = filter_elem
        action_count_elem.output_value_key = "elem_count"

        ctx_elem = GenericCardSystem.resolve_action_with_db(state, action_count_elem, 0, {1000: tamaseed_def}, {})
        self.assertEqual(ctx_elem["elem_count"], 1)

        # 2. Verify "CREATURE" does NOT count it
        filter_creature = FilterDef()
        filter_creature.zones = ["BATTLE_ZONE"]
        filter_creature.types = ["CREATURE"]

        action_count_creature = ActionDef()
        action_count_creature.type = EffectActionType.COUNT_CARDS
        action_count_creature.filter = filter_creature
        action_count_creature.output_value_key = "creature_count"

        ctx_creature = GenericCardSystem.resolve_action_with_db(state, action_count_creature, 0, {1000: tamaseed_def}, {})
        self.assertEqual(ctx_creature["creature_count"], 0)

        # 3. Verify "TAMASEED" counts it
        filter_tamaseed = FilterDef()
        filter_tamaseed.zones = ["BATTLE_ZONE"]
        filter_tamaseed.types = ["TAMASEED"]

        action_count_tamaseed = ActionDef()
        action_count_tamaseed.type = EffectActionType.COUNT_CARDS
        action_count_tamaseed.filter = filter_tamaseed
        action_count_tamaseed.output_value_key = "tamaseed_count"

        ctx_tamaseed = GenericCardSystem.resolve_action_with_db(state, action_count_tamaseed, 0, {1000: tamaseed_def}, {})
        self.assertEqual(ctx_tamaseed["tamaseed_count"], 1)

    def test_element_count_evolution(self):
        # Test that an evolution creature counts as 1 element
        state = GameState(100)

        # Create Evo Creature
        evo_def = CardDefinition()
        evo_def.id = 2000
        evo_def.type = CardType.EVOLUTION_CREATURE

        # Create underlying card
        base_def = CardDefinition()
        base_def.id = 2001
        base_def.type = CardType.CREATURE

        # Add Evo to Battle Zone (Instance 1)
        state.add_test_card_to_battle(0, 2000, 1, False, False)
        # In current engine, underlying cards are stored in `underlying_cards` vector of the instance, or linked.
        # But `add_test_card_to_battle` just puts one instance.
        # This represents an evolution creature stack.

        filter_elem = FilterDef()
        filter_elem.zones = ["BATTLE_ZONE"]
        filter_elem.types = ["ELEMENT"]

        action = ActionDef()
        action.type = EffectActionType.COUNT_CARDS
        action.filter = filter_elem
        action.output_value_key = "count"

        ctx = GenericCardSystem.resolve_action_with_db(state, action, 0, {2000: evo_def, 2001: base_def}, {})
        self.assertEqual(ctx["count"], 1)

if __name__ == '__main__':
    unittest.main()
