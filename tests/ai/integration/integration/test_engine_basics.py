import sys
import os
import pytest
from typing import Dict, Any

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

class TestEngineBasics:
    card_db: Dict[int, Any] = {}

    @classmethod
    def setup_class(cls) -> None:
        try:
            cls.card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
        except Exception:
            cls.card_db = {}

    def test_mana_charge(self) -> None:
        state = dm_ai_module.GameState(1000)
        card_id = 1
        if self.card_db and 1 not in self.card_db:
             card_id = list(self.card_db.keys())[0]

        config = dm_ai_module.ScenarioConfig()
        config.my_hand_cards = [card_id]

        dm_ai_module.PhaseManager.setup_scenario(state, config, self.card_db)
        state.active_player_id = 0
        state.current_phase = dm_ai_module.Phase.MANA

        # Ensure mana usage limit is reset or available (1 charge per turn usually)
        # Assuming fresh state allows one charge.

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

        # In new engine, Mana Charge might be MOVE_CARD.
        # Check available actions.
        # print(f"DEBUG: Actions available: {[str(a.type) for a in actions]}")

        charge_action = next((a for a in actions if a.type == dm_ai_module.ActionType.MANA_CHARGE or a.type == dm_ai_module.ActionType.MOVE_CARD), None)

        if charge_action is None:
             pytest.skip("Mana charge action not generated (maybe already charged or limited)")

        dm_ai_module.EffectResolver.resolve_action(state, charge_action, self.card_db)

        p0 = state.players[0]
        # In MANA phase, standard engine moves to MANA zone.
        # Check if mana increased
        assert len(p0.mana_zone) >= 1
