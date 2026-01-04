#include "core/game_state.hpp"

namespace dm::core {

    void GameState::on_card_play(CardID cid, int turn, bool is_trigger, int cost_diff, PlayerID pid) {
        // Record immediate usage stats
        bool from_hand = true; // Default assumption for play_card
        // If it was a trigger, it came from shield or hand (trigger).
        // on_card_play is usually called by generic_card_system or effect_resolver.
        // We might need to be more specific about source.
        // But the requirement says "usage from shield/hand".
        // is_trigger=true implies Shield Trigger (mostly) or S-Back (Hand).
        // Since S-Trigger is "from shield", and is_trigger covers it.
        // What about "Hand Play"? If !is_trigger, it's a normal play or effect summon.
        // Usually on_card_play represents a deliberate play action.

        get_mutable_card_stats(cid).record_usage(turn, is_trigger, cost_diff, from_hand);
        // Record per-game history for win rate calculation
        if (pid == 0 || pid == 1) {
            played_cards_history_this_game[pid].push_back({cid, turn});
        }
    }

    void GameState::on_game_finished(GameResult result) {
        if (stats_recorded) return;
        stats_recorded = true;

        int winner_id = -1;
        if (result == GameResult::P1_WIN) winner_id = 0;
        else if (result == GameResult::P2_WIN) winner_id = 1;

        // 1. Scan Mana Zones for Resource Usage
        for (const auto& player : players) {
            for (const auto& card : player.mana_zone) {
                CardID cid = card.card_id;
                get_mutable_card_stats(cid).mana_usage_count++;
            }
        }

        if (winner_id != -1) {
            bool is_comeback = (players[winner_id].shield_zone.empty());
            int winning_turn = turn_number;

            // Update stats for cards played by the winner
            for (const auto& entry : played_cards_history_this_game[winner_id]) {
                CardID cid = entry.first;
                int played_turn = entry.second;

                CardStats& stats = get_mutable_card_stats(cid);
                stats.win_count++;
                stats.sum_win_contribution += 1.0;

                if (is_comeback) {
                    stats.sum_comeback_win += 1.0;
                }
                if (played_turn == winning_turn) {
                    stats.sum_finish_blow += 1.0;
                }
            }
        }
    }
}
