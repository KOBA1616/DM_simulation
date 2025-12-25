
from typing import Any, List, Dict, Optional


class HeuristicAgent:
    def __init__(self, player_id: int) -> None:
        self.player_id: int = player_id

    def get_action(self, state: Any, legal_actions: List[Any], card_db: Dict[int, Any]) -> Optional[Any]:
        import dm_ai_module
        import random

        if not legal_actions:
            return None

        def get_def(cid: int) -> Optional[Any]:
            if cid in card_db:
                return card_db[cid]
            return None

        # 1. Mana Charge
        mana_actions = [a for a in legal_actions if a.type == dm_ai_module.ActionType.MANA_CHARGE]
        if mana_actions:
            current_mana = len(state.players[self.player_id].mana_zone)
            if current_mana < 8:
                return random.choice(mana_actions)

        # 2. Play Card
        play_actions = [a for a in legal_actions if a.type == dm_ai_module.ActionType.PLAY_CARD]
        if play_actions:
            def get_cost(action: Any) -> int:
                cdef = get_def(action.card_id)
                return int(cdef.cost) if cdef and hasattr(cdef, 'cost') else 0

            play_actions.sort(key=get_cost, reverse=True)
            return play_actions[0]

        # 3. Attack Player (Aggro)
        attack_player = [a for a in legal_actions if a.type == dm_ai_module.ActionType.ATTACK_PLAYER]
        if attack_player:
            return random.choice(attack_player)

        # 4. Attack Creature
        attack_creature = [a for a in legal_actions if a.type == dm_ai_module.ActionType.ATTACK_CREATURE]
        if attack_creature:
            return random.choice(attack_creature)

        # 5. Block
        block_actions = [a for a in legal_actions if a.type == dm_ai_module.ActionType.BLOCK]
        if block_actions:
            if len(state.players[self.player_id].shield_zone) <= 2:
                return random.choice(block_actions)
            if random.random() < 0.5:
                return random.choice(block_actions)

        # 6. Select Target (for effects)
        select_actions = [a for a in legal_actions if a.type == dm_ai_module.ActionType.SELECT_TARGET]
        if select_actions:
            return random.choice(select_actions)

        # 7. Shield Trigger
        st_actions = [a for a in legal_actions if a.type == dm_ai_module.ActionType.USE_SHIELD_TRIGGER]
        if st_actions:
            return st_actions[0]

        return random.choice(legal_actions)
