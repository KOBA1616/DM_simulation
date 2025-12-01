#include "core/game_state.hpp"

namespace dm::core {

    void GameState::on_card_play(CardID cid, int turn, bool is_trigger, int cost_diff, PlayerID pid) {
        // Record immediate usage stats
        auto it = global_card_stats.find(cid);
        if (it != global_card_stats.end()) {
            it->second.record_usage(turn, is_trigger, cost_diff);
        } else {
             CardStats cs;
             cs.record_usage(turn, is_trigger, cost_diff);
             global_card_stats[cid] = cs;
        }
        // Record per-game history for win rate calculation
        if (pid == 0 || pid == 1) {
            played_cards_history_this_game[pid].push_back(cid);
        }
    }

    void GameState::on_game_finished(GameResult result) {
        int winner_id = -1;
        if (result == GameResult::P1_WIN) winner_id = 0;
        else if (result == GameResult::P2_WIN) winner_id = 1;

        if (winner_id != -1) {
            // Update stats for cards played by the winner
            for (CardID cid : played_cards_history_this_game[winner_id]) {
                auto it = global_card_stats.find(cid);
                if (it != global_card_stats.end()) {
                    it->second.win_count++;
                    it->second.sum_win_contribution += 1.0;
                }
            }
            // Optional: Track loss stats or differential?
            // Currently sum_win_contribution accumulates 1.0 for wins, 0.0 for losses (implicit by not adding).
            // So avg = sum_win_contribution / play_count will be the win rate.
        }
    }
}
