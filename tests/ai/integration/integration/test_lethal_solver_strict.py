
import pytest
import dm_ai_module
from dm_ai_module import GameState, CardDefinition, Civilization, CardType, CardKeywords, PassiveEffect, PassiveType, FilterDef

class TestLethalSolverStrict:
    def setup_method(self):
        self.game = GameState(100)
        self.card_db = {}
        self.game.active_player_id = 0
        self.game.players[0].mana_zone = []
        self.game.players[1].shield_zone = [] # No shields for simple lethal

    def register_card(self, cid, power=3000, is_blocker=False, is_sa=True, passives=None):
        c = CardDefinition()
        c.name = f"Card_{cid}"
        c.power = power
        c.cost = 5
        c.type = CardType.CREATURE
        c.civilizations = [Civilization.FIRE]
        c.keywords = CardKeywords()
        c.keywords.speed_attacker = is_sa
        c.keywords.blocker = is_blocker
        self.card_db[cid] = c
        return c

    def test_unblockable_by_power(self):
        """
        Attacker: 5000 Power, SA. "Can't be blocked by power 4000 or less"
        Blocker: 3000 Power, Blocker.

        Current Solver: Sees 'Blocker' -> Thinks it can block -> Not Lethal.
        Strict Solver: Sees restriction -> Blocker invalid -> Lethal.
        """
        att_id = 1
        blk_id = 2

        att_def = self.register_card(att_id, 5000, is_sa=True)
        blk_def = self.register_card(blk_id, 3000, is_blocker=True, is_sa=False)

        self.game.add_test_card_to_battle(0, att_id, 0, False, True) # Player 0, Attacker
        self.game.add_test_card_to_battle(1, blk_id, 1, False, False) # Player 1, Blocker (Untapped)

        # Add passive: Blocker (ID 1) cannot block.
        passive = PassiveEffect()
        passive.type = PassiveType.CANNOT_BLOCK
        passive.target_filter = FilterDef()
        passive.target_filter.min_power = 3000
        passive.target_filter.max_power = 3000
        passive.controller = 1
        passive.source_instance_id = 1

        # Use helper instead of property setter
        self.game.add_passive_effect(passive)

        print(f"DEBUG: Passives count in python: {len(self.game.passive_effects)}")

        is_lethal = dm_ai_module.LethalSolver.is_lethal(self.game, self.card_db)
        assert is_lethal == True, "Should be lethal because blocker cannot block due to passive"
