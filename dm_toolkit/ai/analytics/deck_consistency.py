import dm_ai_module
import random
import logging
from typing import Any, List
from dm_toolkit.types import CardDB, SeenCards

class SolitaireRunner:
    """
    Runs a solitaire (1-player) simulation to measure deck consistency.
    Corresponds to Requirement 2-A: Access Count Simulation.
    """
    def __init__(self, card_db: CardDB, deck_list: List[int], max_turns: int = 5) -> None:
        self.card_db: CardDB = card_db
        self.deck_list: List[int] = deck_list
        self.max_turns: int = max_turns
        self.logger = logging.getLogger("SolitaireRunner")

    def run_simulation(self) -> int:
        """
        Runs one simulation and returns the number of unique cards accessed.
        """
        # 1. Setup State
        state = dm_ai_module.GameState(1000)

        instance_id_counter = 0

        # Add cards to Player 0 Deck
        for card_id in self.deck_list:
            state.add_card_to_deck(0, card_id, instance_id_counter)
            instance_id_counter += 1

        # Add dummy cards to Player 1 Deck
        for _ in range(30):
            state.add_card_to_deck(1, 1, instance_id_counter)
            instance_id_counter += 1

        # 3. Start Game
        dm_ai_module.PhaseManager.start_game(state, self.card_db)

        seen_cards: SeenCards = set()

        # 4. Game Loop
        for turn in range(1, self.max_turns + 1):
            if state.game_over:
                break

            current_turn = state.turn_number
            # Loop until turn changes or game over
            turn_action_count = 0
            MAX_ACTIONS = 1000 # Safety break

            while state.turn_number == current_turn and not state.game_over:
                turn_action_count += 1
                if turn_action_count > MAX_ACTIONS:
                    print(f"WARNING: Max actions reached for turn {current_turn}. Breaking.")
                    break

                player_id = state.active_player_id

                if player_id == 0:
                    self._play_turn_heuristic(state)
                else:
                    self._pass_turn(state)

            # End of Turn Scan
            self._scan_accessed_cards(state, 0, seen_cards)

        return len(seen_cards)

    def _play_turn_heuristic(self, state: Any) -> None:
        """
        Simple heuristic: Mana Charge -> Play max cost card -> Attack -> End.
        """
        # We process ONE action per call to this function in the main loop?
        # No, the main loop calls this, so this function should return after making progress.
        # But wait, the main loop checks 'active_player_id'.
        # So this function should perform actions until active_player_id changes OR return control to check limits.

        # Actually, let's make this function perform ONE step,
        # and let the main loop handle the 'while active_player_id == 0'.
        # Ah, the previous implementation had a `while` loop inside `_play_turn_heuristic`.
        # I removed it in the outer loop, so I should put it back or change structure.

        # Correct structure:
        # Main loop checks turn number.
        # Inside: check active player. Call play/pass.
        # Play/Pass should perform ONE action or a sequence that ends.

        # Let's perform ONE action here to respect the outer safety counter.

        actions: List[Any] = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db) or []
        try:
            from dm_toolkit.commands_new import generate_legal_commands
        except Exception:
            generate_legal_commands = None
        cmds = generate_legal_commands(state, self.card_db) if generate_legal_commands else []

        if not actions and not cmds:
            dm_ai_module.PhaseManager.next_phase(state, self.card_db)
            return

        best_action: Any = self._choose_action(actions, state) if actions else None
        best_cmd = cmds[0] if cmds else None

        # Prefer executing command when available
        if best_cmd is not None:
            try:
                state.execute_command(best_cmd)
            except Exception:
                try:
                    best_cmd.execute(state)
                except Exception:
                    if best_action is not None:
                        dm_ai_module.EffectResolver.resolve_action(state, best_action, self.card_db)
        else:
            if best_action is None:
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)
            else:
                if best_action.type == dm_ai_module.ActionType.PASS:
                    dm_ai_module.PhaseManager.next_phase(state, self.card_db)
                else:
                    dm_ai_module.EffectResolver.resolve_action(state, best_action, self.card_db)

    def _choose_action(self, actions: List[Any], state: Any) -> Any:
        # Prioritize:
        # 1. Mana Charge (if not done)
        # 2. Play Card (Effect)
        # 3. Attack (Player)
        # 4. Pass

        charge_actions = [a for a in actions if a.type == dm_ai_module.ActionType.MANA_CHARGE]
        play_actions = [a for a in actions if a.type == dm_ai_module.ActionType.PLAY_CARD]
        attack_actions = [a for a in actions if a.type == dm_ai_module.ActionType.ATTACK_PLAYER or a.type == dm_ai_module.ActionType.ATTACK_CREATURE]
        pass_actions = [a for a in actions if a.type == dm_ai_module.ActionType.PASS]

        if charge_actions:
            return random.choice(charge_actions)

        if play_actions:
            return max(play_actions, key=lambda a: self.card_db[state.get_card_instance(a.source_instance_id).card_id].cost)

        if attack_actions:
            return random.choice(attack_actions)

        if pass_actions:
            return pass_actions[0]

        return actions[0] # Fallback

    def _pass_turn(self, state: Any) -> None:
        """Pass through opponent turn."""
        # Perform one step for opponent
        actions: List[Any] = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)
        if not actions:
             dm_ai_module.PhaseManager.next_phase(state, self.card_db)
             return

        pass_action: Any = next((a for a in actions if a.type == dm_ai_module.ActionType.PASS), None)

        if pass_action:
             dm_ai_module.PhaseManager.next_phase(state, self.card_db)
        else:
            # Must perform mandatory action (e.g. resolve effect)
             dm_ai_module.EffectResolver.resolve_action(state, actions[0], self.card_db)

    def _scan_accessed_cards(self, state: Any, player_id: int, seen_cards: SeenCards) -> None:
        # Scan Hand
        for card in state.players[player_id].hand:
            seen_cards.add(card.instance_id)

        # Scan Mana
        for card in state.players[player_id].mana_zone:
            seen_cards.add(card.instance_id)

        # Scan Battle Zone
        for card in state.players[player_id].battle_zone:
            seen_cards.add(card.instance_id)

        # Scan Graveyard
        for card in state.players[player_id].graveyard:
            seen_cards.add(card.instance_id)
