
import sys
import os
import pytest
import dm_ai_module
from dm_ai_module import (
    CardData, CardType, Civilization, FilterDef, ActionDef, EffectDef,
    TriggerType, ActionType, EffectActionType, ConditionDef, TargetScope,
    GameState, Phase, ActionGenerator, EffectResolver, CardRegistry,
    GenericCardSystem
)

# Helper to register a card
def register_test_card(card_id: int, name: str, effects: list, cost: int = 5, civ: str = "FIRE"):
    card = CardData(card_id, name, cost, civ, 5000, "CREATURE", [], effects)
    dm_ai_module.register_card_data(card)
    return card

def process_pending(state, card_db):
    for _ in range(20):
        if not state.pending_effects:
            break
        actions = ActionGenerator.generate_legal_actions(state, card_db)
        if not actions:
            break
        target_action = None
        for a in actions:
            if a.type == ActionType.RESOLVE_EFFECT:
                target_action = a
                break
        if not target_action:
            for a in actions:
                if a.type == ActionType.SELECT_TARGET:
                    target_action = a
                    break
        if target_action:
            EffectResolver.resolve_action(state, target_action, card_db)
        else:
            break

class TestAtomicActionVersatility:
    @classmethod
    def setup_class(cls):
        cls.card_db = {}

    def setup_method(self):
        pass

    def test_case_1_sequential_dependency(self):
        # Action 1: Destroy self
        action1 = ActionDef()
        action1.type = EffectActionType.DESTROY
        action1.scope = TargetScope.SELF
        action1.value1 = 1
        action1.output_value_key = "destroyed_count"

        # Action 2: Draw 2 (Conditional)
        action2 = ActionDef()
        action2.type = EffectActionType.DRAW_CARD
        action2.value1 = 2

        cond = ConditionDef()
        cond.type = "COMPARE_STAT"
        cond.stat_key = "destroyed_count"
        cond.op = ">="
        cond.value = 1
        action2.condition = cond

        effect = EffectDef()
        effect.trigger = TriggerType.ON_PLAY
        effect.actions = [action1, action2]

        card_id = 10001
        register_test_card(card_id, "SelfSacrificeDraw", [effect])

        state = GameState(100)
        state.players[0].hand = []
        state.players[0].battle_zone = []
        state.players[0].graveyard = []

        state.add_test_card_to_battle(0, card_id, 0, False, False)

        GenericCardSystem.resolve_effect(state, effect, 0)
        process_pending(state, self.card_db)

        # Even if it fails, we document the state
        # Expectation: destroyed
        # If not destroyed, it means engine didn't process destroy correctly in this context
        assert len(state.players[0].battle_zone) == 0, "Card should be destroyed"
        assert len(state.players[0].graveyard) == 1, "Card should be in graveyard"
        assert len(state.players[0].hand) == 2, "Should draw 2 cards"

    def test_case_2_conditional_trigger(self):
        act1 = ActionDef()
        act1.type = EffectActionType.COUNT_CARDS
        act1.output_value_key = "fire_mana_count"
        f = FilterDef()
        f.zones = ["MANA_ZONE"]
        f.civilizations = [Civilization.FIRE]
        f.owner = "SELF"
        act1.filter = f

        act2 = ActionDef()
        act2.type = EffectActionType.BREAK_SHIELD
        act2.value1 = 1
        # Explicitly set filter for BREAK_SHIELD to ensure it finds targets
        act2.filter.zones = ["SHIELD_ZONE"]
        act2.filter.owner = "OPPONENT"

        cond = ConditionDef()
        cond.type = "COMPARE_STAT"
        cond.stat_key = "fire_mana_count"
        cond.op = ">="
        cond.value = 3
        act2.condition = cond

        effect = EffectDef()
        effect.trigger = TriggerType.ON_ATTACK
        effect.actions = [act1, act2]

        card_id = 10002
        register_test_card(card_id, "FireBreaker", [effect])

        state = GameState(100)
        state.players[0].hand = []
        state.add_test_card_to_battle(0, card_id, 0, False, False)

        dummy_fire = 9999
        register_test_card(dummy_fire, "FireMana", [], 1, "FIRE")
        state.add_card_to_mana(0, dummy_fire, 1)
        state.add_card_to_mana(0, dummy_fire, 2)

        # Setup Shields explicitly
        for i in range(3):
            state.add_card_to_deck(1, dummy_fire, 100+i)
            setup_act = ActionDef()
            setup_act.type = EffectActionType.ADD_SHIELD
            setup_act.value1 = 1
            GenericCardSystem.resolve_action(state, setup_act, 100+i)
            process_pending(state, self.card_db)

        # Resolve Effect (2 Mana < 3)
        GenericCardSystem.resolve_effect(state, effect, 0)
        process_pending(state, self.card_db)
        assert len(state.players[1].shield_zone) == 3 # No break

        # Add 3rd Mana
        state.add_card_to_mana(0, dummy_fire, 3)

        # Resolve again
        GenericCardSystem.resolve_effect(state, effect, 0)
        process_pending(state, self.card_db)
        assert len(state.players[1].shield_zone) == 2, "Should break 1 shield"

    def test_case_3_variable_linking_draw(self):
        act1 = ActionDef()
        act1.type = EffectActionType.COUNT_CARDS
        act1.output_value_key = "shield_count"
        f = FilterDef()
        f.zones = ["SHIELD_ZONE"]
        f.owner = "SELF"
        act1.filter = f

        act2 = ActionDef()
        act2.type = EffectActionType.DRAW_CARD
        act2.input_value_key = "shield_count"
        # Set value1=1 as fallback. If it draws 1, linking failed. If 4, success.
        act2.value1 = 1

        effect = EffectDef()
        effect.trigger = TriggerType.ON_PLAY
        effect.actions = [act1, act2]

        card_id = 10003
        register_test_card(card_id, "ShieldDraw", [effect])

        state = GameState(100)
        state.players[0].hand = []

        for i in range(4):
            state.add_card_to_deck(0, 9999, 200+i)
            setup_act = ActionDef()
            setup_act.type = EffectActionType.ADD_SHIELD
            setup_act.value1 = 1
            GenericCardSystem.resolve_action(state, setup_act, 200+i)
            process_pending(state, self.card_db)

        assert len(state.players[0].shield_zone) == 4

        state.add_test_card_to_battle(0, card_id, 0, False, False)

        GenericCardSystem.resolve_effect(state, effect, 0)
        process_pending(state, self.card_db)

        assert len(state.players[0].hand) == 4, f"Should draw 4 cards, but drew {len(state.players[0].hand)}"

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
