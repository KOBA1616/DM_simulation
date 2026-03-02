
from typing import Any, List, Dict, Optional


class HeuristicAgent:
    def __init__(self, player_id: int) -> None:
        self.player_id: int = player_id

    def get_command(self, state: Any, legal_commands: List[Any], card_db: Dict[int, Any]) -> Optional[Any]:
        import dm_ai_module
        import random

        # 再発防止: dm_ai_module.CommandType を使用。ActionType は C++ レガシースタブ。
        CommandType = getattr(dm_ai_module, 'CommandType', None) or getattr(dm_ai_module, 'ActionType', None)

        def _type_is(cmd_obj: Any, name: str) -> bool:
            try:
                t = getattr(cmd_obj, 'type', None)
                if CommandType is not None and hasattr(CommandType, name):
                    return bool(t == getattr(CommandType, name))
                # Fallbacks: compare by enum name or string match
                if isinstance(t, str) and t.upper().endswith(name):
                    return True
                try:
                    return bool(getattr(t, 'name', '').upper() == name)
                except Exception:
                    return False
            except Exception:
                return False

        if not legal_commands:
            return None

        def get_def(cid: int) -> Optional[Any]:
            if cid in card_db:
                return card_db[cid]
            return None

        # 1. Mana Charge
        mana_cmds = [c for c in legal_commands if _type_is(c, 'MANA_CHARGE')]
        if mana_cmds:
            current_mana = len(state.players[self.player_id].mana_zone)
            if current_mana < 8:
                return random.choice(mana_cmds)

        # 2. Play Card
        play_cmds = [c for c in legal_commands if _type_is(c, 'PLAY_CARD') or _type_is(c, 'DECLARE_PLAY') or _type_is(c, 'PLAY_FROM_ZONE')]
        if play_cmds:
            def get_cost(cmd: Any) -> int:
                cdef = get_def(getattr(cmd, 'card_id', 0))
                return int(cdef.cost) if cdef and hasattr(cdef, 'cost') else 0

            play_cmds.sort(key=get_cost, reverse=True)
            return play_cmds[0]

        # 3. Attack Player (Aggro)
        attack_player_cmds = [c for c in legal_commands if _type_is(c, 'ATTACK_PLAYER')]
        if attack_player_cmds:
            return random.choice(attack_player_cmds)

        # 4. Attack Creature
        attack_creature_cmds = [c for c in legal_commands if _type_is(c, 'ATTACK_CREATURE')]
        if attack_creature_cmds:
            return random.choice(attack_creature_cmds)

        # 5. Block
        block_cmds = [c for c in legal_commands if _type_is(c, 'BLOCK')]
        if block_cmds:
            if len(state.players[self.player_id].shield_zone) <= 2:
                return random.choice(block_cmds)
            if random.random() < 0.5:
                return random.choice(block_cmds)

        # 6. Select Target (for effects)
        select_cmds = [c for c in legal_commands if _type_is(c, 'SELECT_TARGET')]
        if select_cmds:
            return random.choice(select_cmds)

        # 7. Shield Trigger
        st_cmds = [c for c in legal_commands if _type_is(c, 'USE_SHIELD_TRIGGER')]
        if st_cmds:
            return st_cmds[0]

        return random.choice(legal_commands)

    # 後方互換エイリアス。新規コードでは get_command を使用すること。
    get_action = get_command
