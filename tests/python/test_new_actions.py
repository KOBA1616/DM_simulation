import unittest
import dm_ai_module
from dm_ai_module import GameState, ActionDef, EffectActionType, EffectDef, CardDefinition, TriggerType, CardData, CardKeywords, ConditionDef, GenericCardSystem

class TestNewActions(unittest.TestCase):
    def setUp(self):
        self.state = GameState(100)
        self.state.setup_test_duel()

        # Register a test spell and test creature
        self.spell_data = CardData(
            1000,           # id
            "Test Spell",   # name
            0,              # cost
            "FIRE",         # civ
            0,              # power
            "SPELL",        # type
            [],             # races
            []              # effects
        )
        dm_ai_module.register_card_data(self.spell_data)

        self.creature_data = CardData(
            1001,
            "Test Creature",
            0,
            "NATURE",
            1000,
            "CREATURE",
            [],
            []
        )
        dm_ai_module.register_card_data(self.creature_data)

    def test_cast_spell_action(self):
        # Setup: Add spell to hand
        self.state.add_card_to_hand(0, 1000, 1) # Player 0, Card 1000, Instance 1

        # Verify it's in hand
        hand = self.state.players[0].hand
        self.assertEqual(len(hand), 1)
        self.assertEqual(hand[0].instance_id, 1)

        # Create action def: Cast from hand
        action = ActionDef()
        action.type = EffectActionType.CAST_SPELL
        action.filter.zones = ["HAND"]
        action.filter.owner = "SELF" # Implied active player 0

        # We need to manually invoke resolve_action on the GenericCardSystem
        # Since the handler relies on ctx.targets, we need to pass targets or filter must match.
        # But ActionDef execution usually involves target selection IF filters are used.
        # However, for unit testing the handler logic directly, we might need resolve_with_targets
        # OR we can let GenericCardSystem handle selection if we mock user input? No.

        # Let's assume we pass the target directly to test logic.
        # The handler is registered.

        # But wait, GenericCardSystem.resolve_action calls handler->resolve().
        # CastSpellHandler.resolve() is empty! It relies on resolve_with_targets.
        # This implies CAST_SPELL is designed to be used after a SELECT_TARGET action.

        # So we should call resolve_effect_with_targets.

        effect = EffectDef(
            TriggerType.NONE,
            ConditionDef(),
            [action]
        )

        targets = [1] # The instance ID

        # Resolve
        GenericCardSystem.resolve_effect_with_targets(
            self.state,
            effect,
            targets,
            -1, # Source ID (n/a)
            {}, # CardDB (empty, uses registry fallback or we need to pass a map)
            {}  # Context
        )

        # Assertions
        # 1. Card should be removed from hand
        hand = self.state.players[0].hand
        self.assertEqual(len(hand), 0)

        # 2. Card should be in Stack (or resolved to Grave).
        # Since it's a spell with 0 effects, resolve_play_from_stack moves it to Stack, resolves (nothing), then moves to Grave.
        grave = self.state.players[0].graveyard
        self.assertEqual(len(grave), 1)
        self.assertEqual(grave[0].instance_id, 1)

    def test_put_creature_action(self):
        # Setup: Add creature to hand
        self.state.add_card_to_hand(0, 1001, 2)

        # Action
        action = ActionDef()
        action.type = EffectActionType.PUT_CREATURE

        effect = EffectDef(
            TriggerType.NONE,
            ConditionDef(),
            [action]
        )

        targets = [2]

        GenericCardSystem.resolve_effect_with_targets(
            self.state,
            effect,
            targets,
            -1,
            {},
            {}
        )

        # Assertions
        # 1. Removed from hand
        hand = self.state.players[0].hand
        self.assertEqual(len(hand), 0)

        # 2. Added to Battle Zone
        bz = self.state.players[0].battle_zone
        self.assertEqual(len(bz), 1)
        self.assertEqual(bz[0].instance_id, 2)
        self.assertTrue(bz[0].summoning_sickness)

if __name__ == '__main__':
    unittest.main()
