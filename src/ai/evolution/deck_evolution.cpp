#include "deck_evolution.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

namespace dm::ai {

    using namespace dm::core;

    DeckEvolution::DeckEvolution(const std::map<CardID, CardDefinition>& card_db)
        : card_db_(card_db) {
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    std::vector<int> DeckEvolution::evolve_deck(const std::vector<int>& current_deck,
                                              const std::vector<int>& candidate_pool,
                                              const DeckEvolutionConfig& config) {
        std::vector<int> new_deck = current_deck;

        // Simple mutation: replace N cards
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Count how many to mutate
        int mutations = 0;
        for (size_t i = 0; i < new_deck.size(); ++i) {
            if (dist(rng_) < config.mutation_rate) {
                mutations++;
            }
        }

        if (mutations == 0 && !candidate_pool.empty()) mutations = 1; // Ensure at least one change if possible

        std::uniform_int_distribution<int> pool_dist(0, candidate_pool.size() - 1);
        std::uniform_int_distribution<int> deck_dist(0, new_deck.size() - 1);

        for (int i = 0; i < mutations; ++i) {
            if (candidate_pool.empty()) break;

            int deck_idx = deck_dist(rng_);
            int pool_idx = pool_dist(rng_);

            // Basic check: maintain civilization consistency?
            // For now, random replacement. The "Evaluation" step (Game) determines fitness.
            // But we can try to be smarter: replace with same color if possible.

            // Get card to be removed
            CardID old_id = static_cast<CardID>(new_deck[deck_idx]);
            // CardID new_id = static_cast<CardID>(candidate_pool[pool_idx]);

            // Simple replacement
            new_deck[deck_idx] = candidate_pool[pool_idx];
        }

        // Enforce deck size constraint if needed (though input should be valid)
        if ((int)new_deck.size() > config.target_deck_size) {
            new_deck.resize(config.target_deck_size);
        }

        return new_deck;
    }

    float DeckEvolution::get_pair_synergy(const CardDefinition& c1, const CardDefinition& c2) {
        float score = 0.0f;

        // 1. Civilization Match
        bool civ_match = false;
        for (auto civ1 : c1.civilizations) {
            if (c1.has_civilization(civ1) && c2.has_civilization(civ1)) {
                civ_match = true;
                break;
            }
        }
        if (civ_match) score += 1.0f;

        // 2. Race Match (Evolution Base)
        bool race_match = false;
        for (const auto& r1 : c1.races) {
            for (const auto& r2 : c2.races) {
                if (r1 == r2) {
                    race_match = true;
                    break;
                }
            }
        }
        if (race_match) score += 0.5f;

        // 3. Evolution Synergy
        // If c1 is evolution and c2 has matching race -> High Score
        if (c1.keywords.evolution) {
            for (const auto& r2 : c2.races) {
                 // Simplified check: usually race match is enough.
                 // Ideally check if c1 evolves FROM r2.
                 // But we don't strictly have "evolves from" data parsed easily here without parsing text/filter.
                 // Assuming race match is a good proxy.
                 // Also specific races like "Dragon" often evolve into dragons.
                 for (const auto& r1 : c1.races) {
                     if (r1 == r2) {
                         score += 2.0f;
                     }
                 }
            }
        }

        // 4. Cost Curve (Speed Attacker + Low Cost?)
        // (Handled by curve score mostly)

        return score;
    }

    float DeckEvolution::calculate_interaction_score(const std::vector<int>& deck_ids) {
        if (deck_ids.empty()) return 0.0f;

        float total_synergy = 0.0f;
        int pair_count = 0;

        // Sample pairs or full N^2? N=40 -> 1600 checks, fast enough in C++.
        for (size_t i = 0; i < deck_ids.size(); ++i) {
            if (card_db_.find(static_cast<CardID>(deck_ids[i])) == card_db_.end()) continue;
            const auto& c1 = card_db_.at(static_cast<CardID>(deck_ids[i]));

            for (size_t j = i + 1; j < deck_ids.size(); ++j) {
                if (card_db_.find(static_cast<CardID>(deck_ids[j])) == card_db_.end()) continue;
                const auto& c2 = card_db_.at(static_cast<CardID>(deck_ids[j]));

                total_synergy += get_pair_synergy(c1, c2);
                pair_count++;
            }
        }

        return (pair_count > 0) ? (total_synergy / pair_count) : 0.0f;
    }

    std::vector<int> DeckEvolution::get_candidates_by_civ(const std::vector<int>& pool, Civilization civ) {
        std::vector<int> result;
        for (int id : pool) {
            if (card_db_.count(static_cast<CardID>(id))) {
                if (card_db_.at(static_cast<CardID>(id)).has_civilization(civ)) {
                    result.push_back(id);
                }
            }
        }
        return result;
    }

}
