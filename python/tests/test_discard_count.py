
import unittest
from dm_ai_module import (
    GameState, CardDefinition, EffectActionType, TargetScope,
    ActionDef, EffectDef, TriggerType, FilterDef, Zone, GameInstance,
    EffectSystem, Civilization, CardData
)

class TestDiscardCount(unittest.TestCase):
    def test_discard_output_count_handler(self):
        # 1. Setup GameState
        game = GameState(100)

        # Add cards to player 0's hand
        game.add_card_to_hand(0, 10, 0)
        game.add_card_to_hand(0, 11, 1)
        game.add_card_to_hand(0, 12, 2)

        # Define Action
        action = ActionDef()
        action.type = EffectActionType.DISCARD
        action.scope = TargetScope.TARGET_SELECT
        action.output_value_key = "discarded_num"
        action.destination_zone = "GRAVEYARD" # Default implicit

        # Create context
        card_db = {}
        # We need actual card definitions in DB for the target cards
        card_db[10] = CardDefinition()
        card_db[11] = CardDefinition()
        card_db[12] = CardDefinition()

        targets = [0, 1] # Instance IDs

        # We need to call resolve_effect_with_targets
        # Construct an EffectDef containing this action
        effect = EffectDef()
        effect.actions = [action]

        # Invoke EffectSystem
        try:
            # Note: resolve_effect_with_targets signature might vary in bindings
            # The memory says it returns updated context.
            ctx = EffectSystem.resolve_effect_with_targets(
                game, effect, targets, 999, card_db, {}
            )
        except Exception as e:
            print(f"Error calling resolve: {e}")
            ctx = {}

        # 2. Assertions
        print(f"Hand size: {len(game.players[0].hand)}")
        print(f"Graveyard size: {len(game.players[0].graveyard)}")

        # Check if count is in context
        if isinstance(ctx, dict) and "discarded_num" in ctx:
            print(f"Count returned: {ctx['discarded_num']}")
            self.assertEqual(ctx["discarded_num"], 2)
        else:
            print("Count NOT returned")
            self.fail("Discard count not found in context")

if __name__ == '__main__':
    unittest.main()
