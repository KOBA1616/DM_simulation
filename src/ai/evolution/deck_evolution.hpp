#pragma once

#include "core/card_stats.hpp"
#include "core/card_def.hpp"
#include "core/game_state.hpp"
#include <vector>
#include <map>
#include <algorithm>
#include <random>

namespace dm::ai {

    class DeckEvolution {
    public:
        // Configuration parameters
        float active_use_score = 5.0f;
        float resource_use_score = 2.0f;
        float win_contribution_weight = 1.0f; // Weight for the win contribution stat

        // Constructor
        DeckEvolution() = default;

        /**
         * Calculates the interaction score for a single card based on its stats.
         * Formula: (play_count * active_use_score + mana_usage * resource_use_score) / (appearance_count) + win_contribution * weight
         * Note: Since CardStats doesn't track appearance count directly (it tracks play_count),
         * we might need to rely on `play_count` + `mana_usage` (if we add it) or assume appearance count is approximated.
         *
         * However, looking at the requirement: "normalized by appearance count".
         * CardStats currently doesn't track "drawn count" or "in hand count".
         * It tracks "play_count".
         *
         * Let's update CardStats to track "drawn_count" or "mana_source_count" first.
         * For now, we will use a simplified formula using available stats.
         */
        float calculate_score(const dm::core::CardStats& stats) const;

        /**
         * Evolves a deck by removing low-scoring cards and adding cards from the candidate pool.
         *
         * @param current_deck The current list of card IDs in the deck.
         * @param card_stats Map of CardID to CardStats containing performance data.
         * @param candidate_pool List of CardIDs available to be added.
         * @param fixed_cards List of CardIDs that cannot be removed.
         * @param num_changes Number of cards to swap.
         * @param rng Random number generator.
         * @return The new deck list.
         */
        std::vector<dm::core::CardID> evolve_deck(
            const std::vector<dm::core::CardID>& current_deck,
            const std::map<dm::core::CardID, dm::core::CardStats>& card_stats,
            const std::vector<dm::core::CardID>& candidate_pool,
            const std::vector<dm::core::CardID>& fixed_cards,
            int num_changes,
            std::mt19937& rng
        );
    };

}
