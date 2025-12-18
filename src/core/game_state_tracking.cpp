#include "core/game_state.hpp"
#include "engine/game_command/game_command.hpp"

namespace dm::core {

    void GameState::execute_command(std::shared_ptr<dm::engine::game_command::GameCommand> cmd) {
        if (!cmd) return;
        cmd->execute(*this);
        command_history.push_back(cmd);
    }

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
            played_cards_history_this_game[pid].push_back({cid, turn});
        }
    }

    void GameState::on_game_finished(GameResult result) {
        int winner_id = -1;
        if (result == GameResult::P1_WIN) winner_id = 0;
        else if (result == GameResult::P2_WIN) winner_id = 1;

        if (winner_id != -1) {
            bool is_comeback = (players[winner_id].shield_zone.empty());
            int winning_turn = turn_number;

            // Update stats for cards played by the winner
            for (const auto& entry : played_cards_history_this_game[winner_id]) {
                CardID cid = entry.first;
                int played_turn = entry.second;

                auto it = global_card_stats.find(cid);
                if (it != global_card_stats.end()) {
                    it->second.win_count++;
                    it->second.sum_win_contribution += 1.0;

                    if (is_comeback) {
                        it->second.sum_comeback_win += 1.0;
                    }
                    if (played_turn == winning_turn) {
                        it->second.sum_finish_blow += 1.0;
                    }
                }
            }
            // Optional: Track loss stats or differential?
            // Currently sum_win_contribution accumulates 1.0 for wins, 0.0 for losses (implicit by not adding).
            // So avg = sum_win_contribution / play_count will be the win rate.
        }
    }
}
