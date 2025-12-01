#include "core/game_state.hpp"

namespace dm::core {

    void GameState::on_card_play(CardID cid, int turn, bool is_trigger, int cost_diff) {
        auto it = global_card_stats.find(cid);
        if (it != global_card_stats.end()) {
            it->second.record_usage(turn, is_trigger, cost_diff);
        } else {
             // Should not happen if initialized correctly, but safe to ignore or create?
             // Better to create if missing
             CardStats cs;
             cs.record_usage(turn, is_trigger, cost_diff);
             global_card_stats[cid] = cs;
        }
    }

    void GameState::on_game_finished(GameResult result) {
        // We can't easily know which cards contributed to win here without tracking played cards per player
        // For now, this function is a placeholder or needs to iterate over players decks if we want to track "Deck Win Rate"
        // But CardStats.win_count is usually "Win count when included in deck" or "Win count when played"?
        // Spec 15 says "Win Rate: Usage win rate deviation". This implies "When used".
        // So we need to track which cards were USED by the winner.
        // We don't have a "Used Cards" list readily available here unless we track it.
        // I'll leave this empty for now and rely on Python to call on_game_finished with specific cards or
        // implement a "Cards Played by Player" tracker in GameState.
    }
}
