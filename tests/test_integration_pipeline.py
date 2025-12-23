import pytest
import dm_ai_module
from dm_ai_module import GameState, CardDefinition, CardData, EffectActionType, ActionDef, EffectDef, TriggerType, FilterDef, TargetScope, Civilization

# Helper to register test cards
def register_card(card_id, actions):
    # actions is a list of ActionDef dictionaries
    effects = []
    if actions:
        ed = EffectDef()
        ed.trigger = TriggerType.ON_PLAY
        action_list = []
        for a in actions:
            ad = ActionDef()
            ad.type = a.get("type")
            if "scope" in a: ad.scope = a["scope"]
            if "value1" in a: ad.value1 = a["value1"]
            if "input" in a: ad.input_value_key = a["input"]
            if "output" in a: ad.output_value_key = a["output"]
            if "filter" in a:
                fd = FilterDef()
                if "zones" in a["filter"]: fd.zones = a["filter"]["zones"]
                if "count" in a["filter"]: fd.count = a["filter"]["count"]
                ad.filter = fd
            if "zones" in a: # Search destination
                ad.destination_zone = a["zones"]
            action_list.append(ad)

        # FIX: Assign list to ed.actions
        ed.actions = action_list
        effects.append(ed)

    # Use CardData for registration
    cd = CardData(card_id, "TestCard", 1, "FIRE", 1000, "CREATURE", [], effects)
    dm_ai_module.register_card_data(cd)
    return cd

class TestIntegrationPipeline:

    def setup_method(self):
        self.state = GameState(100)
        self.card_db = {} # Local db, but we prefer Registry for tests using register_card
        # Ensure registry is clean/ready or use ids that don't conflict

    def test_pipeline_loop_mana(self):
        # Test Case: "Put top 2 cards of deck into Mana"
        # Handler: ManaHandler (ADD_MANA) -> compiled to loop/repeat

        cid = 1001
        actions = [{
            "type": EffectActionType.ADD_MANA,
            "value1": 2 # Count
        }]
        register_card(cid, actions)

        ed = EffectDef()
        ad = ActionDef()
        ad.type = EffectActionType.ADD_MANA
        ad.value1 = 2

        # FIX: Use assignment instead of append
        ed.actions = [ad]

        p = self.state.players[self.state.active_player_id]
        self.state.add_card_to_deck(self.state.active_player_id, 2001, 1)
        self.state.add_card_to_deck(self.state.active_player_id, 2002, 2)
        initial_mana = len(p.mana_zone)

        # Use resolve_effect via Registry (no db arg)
        dm_ai_module.GenericCardSystem.resolve_effect(self.state, ed, -1)

        assert len(p.mana_zone) == initial_mana + 2

    def test_pipeline_search_variable_link(self):
        # Test Case: Search Deck for 1 card, put to Hand. Then ADD MANA using output count.

        # Register target cards to ensure they are found by SEARCH
        target_id = 2000
        fodder_id = 2001
        dm_ai_module.register_card_data(CardData(target_id, "Target", 1, "WATER", 1000, "CREATURE", [], []))
        dm_ai_module.register_card_data(CardData(fodder_id, "Fodder", 1, "FIRE", 1000, "CREATURE", [], []))

        cid = 1002
        ed = EffectDef()

        # Action 1: Search Water Card (Target)
        # We use specific filter to ensure only 1 match, allowing auto-select optimization to kick in
        # (count 1 >= valid 1) -> No Pause
        ad1 = ActionDef()
        ad1.type = EffectActionType.SEARCH_DECK
        ad1.scope = TargetScope.TARGET_SELECT
        fd = FilterDef()
        fd.zones = ["DECK"]
        fd.civilizations = [Civilization.WATER]
        ad1.filter = fd

        # Action 2: Count Hand
        ad2 = ActionDef()
        ad2.type = EffectActionType.COUNT_CARDS
        fd2 = FilterDef()
        fd2.zones = ["HAND"]
        ad2.filter = fd2
        ad2.output_value_key = "hand_count"

        # Action 3: Add Mana (equal to hand count)
        ad3 = ActionDef()
        ad3.type = EffectActionType.ADD_MANA
        ad3.input_value_key = "hand_count"

        # FIX: Use assignment
        ed.actions = [ad1, ad2, ad3]

        p = self.state.players[self.state.active_player_id]
        # Deck: 1 Target (Water), 5 Fodder (Fire).
        self.state.add_card_to_deck(self.state.active_player_id, target_id, 1)
        self.state.add_card_to_deck(self.state.active_player_id, fodder_id, 5)
        # Hand: 0 cards initially.

        # SEARCH will move 1 (Target) from Deck to Hand. Hand=1.
        # COUNT will find 1.
        # ADD_MANA will move 1 (Fodder) from Deck Top to Mana.

        # Use resolve_effect via Registry. Variable linking is internal to the effect resolution context.
        dm_ai_module.GenericCardSystem.resolve_effect(self.state, ed, -1)

        assert len(p.hand) == 1
        assert len(p.mana_zone) == 1

    def test_destroy_pipeline(self):
        # Test DestroyHandler migration
        # Effect: Destroy all creatures.

        ed = EffectDef()
        ad = ActionDef()
        ad.type = EffectActionType.DESTROY
        fd = FilterDef()
        fd.zones = ["BATTLE_ZONE"]
        ad.filter = fd
        ed.actions.append(ad)

        # FIX: Use assignment
        ed.actions = [ad]

        # Setup with dummy cards (ID 0) to avoid registry requirement for targets
        self.state.add_test_card_to_battle(self.state.active_player_id, 0, 100, False, False)
        self.state.add_test_card_to_battle(self.state.active_player_id, 0, 101, False, False)

        p = self.state.players[self.state.active_player_id]
        assert len(p.battle_zone) == 2

        dm_ai_module.GenericCardSystem.resolve_effect(self.state, ed, -1)

        assert len(p.battle_zone) == 0
        assert len(p.graveyard) == 2

    def test_reveal_pipeline(self):
        # Test RevealHandler
        # Effect: Reveal top 1 card (moves to Buffer).

        ed = EffectDef()
        ad = ActionDef()
        ad.type = EffectActionType.REVEAL_CARDS
        ad.value1 = 1

        # FIX: Use assignment
        ed.actions = [ad]

        self.state.add_card_to_deck(self.state.active_player_id, 0, 200)
        p = self.state.players[self.state.active_player_id]

        dm_ai_module.GenericCardSystem.resolve_effect(self.state, ed, -1)

        assert len(p.deck) == 0

if __name__ == "__main__":
    pytest.main([__file__])
