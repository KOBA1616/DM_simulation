#include "ai/evolution/deck_evolution.hpp"
#include <iostream>
#include <algorithm>
#include <set>

namespace dm::ai {

    float DeckEvolution::calculate_score(const dm::core::CardStats& stats) const {
        // Formula: 5 points for active use, 2 points for resource use.
        // Normalized by total interactions (play + mana).
        // Plus win contribution.

        float active_plays = static_cast<float>(stats.play_count);
        float resource_uses = static_cast<float>(stats.mana_source_count);

        float total_interactions = active_plays + resource_uses;
        if (total_interactions <= 0.0001f) {
            return 0.0f; // Unused card
        }

        float usage_score = (active_plays * active_use_score) + (resource_uses * resource_use_score);

        // Normalize by number of times it was "drawn" or "available".
        // Since we don't track draws exactly, we use total_interactions as a proxy for "times we chose to use it".
        // Wait, if I play it 1 time and mana it 0 times, score is 5 / 1 = 5.
        // If I play it 0 times and mana it 1 time, score is 2 / 1 = 2.
        // This seems correct: playing is better than mana-ing.

        float base_score = usage_score / total_interactions;

        // Add win contribution
        // win_contribution is a sum, so we need average.
        // stats.play_count tracks plays.
        // stats.win_count tracks wins.
        // sum_win_contribution accumulates 1.0 per win.
        // So average win rate when played is sum_win_contribution / play_count.
        // What about when mana-ed?

        // The win contribution metric in CardStats is updated when the card is PLAYED.
        // So it reflects "When played, how often do we win?".

        float win_rate = 0.0f;
        if (stats.play_count > 0) {
            win_rate = static_cast<float>(stats.sum_win_contribution) / static_cast<float>(stats.play_count);
        }

        return base_score + (win_rate * win_contribution_weight);
    }

    std::vector<dm::core::CardID> DeckEvolution::evolve_deck(
        const std::vector<dm::core::CardID>& current_deck,
        const std::map<dm::core::CardID, dm::core::CardStats>& card_stats,
        const std::vector<dm::core::CardID>& candidate_pool,
        const std::vector<dm::core::CardID>& fixed_cards,
        int num_changes,
        std::mt19937& rng
    ) {
        std::vector<dm::core::CardID> new_deck = current_deck;
        std::set<dm::core::CardID> fixed_set(fixed_cards.begin(), fixed_cards.end());

        // 1. Calculate scores for all cards in the deck
        std::vector<std::pair<float, int>> deck_scores; // score, index in new_deck

        for (size_t i = 0; i < new_deck.size(); ++i) {
            dm::core::CardID cid = new_deck[i];

            // Skip fixed cards
            if (fixed_set.find(cid) != fixed_set.end()) {
                continue;
            }

            float score = 0.0f;
            auto it = card_stats.find(cid);
            if (it != card_stats.end()) {
                score = calculate_score(it->second);
            }
            // If no stats, score is 0.

            deck_scores.push_back({score, (int)i});
        }

        // 2. Sort by score ascending (lowest score first)
        std::sort(deck_scores.begin(), deck_scores.end());

        // 3. Remove lowest scoring cards
        // We need to keep indices valid. We'll mark them for removal.
        std::set<int> indices_to_remove;
        int changes_made = 0;

        for (const auto& pair : deck_scores) {
            if (changes_made >= num_changes) break;
            indices_to_remove.insert(pair.second);
            changes_made++;
        }

        // 4. Construct the intermediate deck (without removed cards)
        std::vector<dm::core::CardID> intermediate_deck;
        for (size_t i = 0; i < new_deck.size(); ++i) {
            if (indices_to_remove.find(i) == indices_to_remove.end()) {
                intermediate_deck.push_back(new_deck[i]);
            }
        }

        // 5. Add new cards from candidate pool
        // Simple strategy: Randomly pick from pool.
        // Better strategy: Pick high scoring cards from pool if stats available?
        // For now, random selection as per "Genetic Algorithm" mutation step.
        std::uniform_int_distribution<int> pool_dist(0, candidate_pool.size() - 1);

        for (int i = 0; i < changes_made; ++i) {
            if (candidate_pool.empty()) break;
            int idx = pool_dist(rng);
            intermediate_deck.push_back(candidate_pool[idx]);
        }

        return intermediate_deck;
    }

}
