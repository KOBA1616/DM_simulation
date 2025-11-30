#include "determinizer.hpp"
#include <algorithm>
#include <random>
#include <vector>
#include <chrono>

namespace dm::engine {

    void Determinizer::determinize(dm::core::GameState& state, int observer_player_id) {
        int opponent_id = 1 - observer_player_id;
        auto& opp = state.players[opponent_id];

        // Collect all hidden cards
        std::vector<dm::core::CardInstance> hidden_pool;
        hidden_pool.reserve(opp.hand.size() + opp.deck.size() + opp.shield_zone.size());

        hidden_pool.insert(hidden_pool.end(), opp.hand.begin(), opp.hand.end());
        hidden_pool.insert(hidden_pool.end(), opp.deck.begin(), opp.deck.end());
        hidden_pool.insert(hidden_pool.end(), opp.shield_zone.begin(), opp.shield_zone.end());

        // Shuffle
        // Use a static random engine to avoid re-seeding every time (performance)
        // But for determinization in MCTS, we want variety.
        // MCTS runs in threads? If so, static might be racey.
        // Let's use thread_local.
        static thread_local std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
        
        std::shuffle(hidden_pool.begin(), hidden_pool.end(), rng);

        // Distribute back
        size_t hand_size = opp.hand.size();
        size_t deck_size = opp.deck.size();
        size_t shield_size = opp.shield_zone.size();

        auto it = hidden_pool.begin();

        opp.hand.assign(it, it + hand_size);
        it += hand_size;

        opp.deck.assign(it, it + deck_size);
        it += deck_size;

        opp.shield_zone.assign(it, it + shield_size);
    }

}
