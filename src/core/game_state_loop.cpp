#include "game_state.hpp"

namespace dm::core {

    void GameState::update_loop_check() {
        if (winner != GameResult::NONE) return;

        uint64_t current_hash = calculate_hash();
        int count = 0;
        for (uint64_t past : hash_history) {
            if (past == current_hash) {
                count++;
            }
        }

        // Spec says: "if the same state continues for 3 times" -> so if we find it 2 times in history + current = 3
        if (count >= 2) {
            loop_proven = true;
            // The player demonstrating the loop (active player) wins
            // Or usually loop means draw, but here "give high reward upon loop proof success"
            // So we declare the active player as winner.
            if (active_player_id == 0) {
                winner = GameResult::P1_WIN;
            } else {
                winner = GameResult::P2_WIN;
            }
            // on_game_finished is called by PhaseManager, so we don't call it here to avoid duplication/issues.
        }

        hash_history.push_back(current_hash);
    }
}
