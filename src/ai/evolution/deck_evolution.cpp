#include "deck_evolution.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <map>

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

        if (mutations == 0 && !candidate_pool.empty() && config.mutation_rate > 0) mutations = 1;

        std::uniform_int_distribution<int> pool_dist(0, candidate_pool.size() - 1);
        std::uniform_int_distribution<int> deck_dist(0, new_deck.size() - 1);

        for (int i = 0; i < mutations; ++i) {
            if (candidate_pool.empty()) break;

            int deck_idx = deck_dist(rng_);
            int pool_idx = pool_dist(rng_);

            new_deck[deck_idx] = candidate_pool[pool_idx];
        }

        enforce_constraints(new_deck, candidate_pool);

        return new_deck;
    }

    std::vector<int> DeckEvolution::crossover_decks(const std::vector<int>& deck1,
                                                    const std::vector<int>& deck2,
                                                    const DeckEvolutionConfig& config) {
        std::vector<int> child_deck;
        child_deck.reserve(config.target_deck_size);

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Simple crossover: pick card from deck1 or deck2
        // If sizes differ, this loop logic needs care. Assuming target size.
        int max_len = std::max(deck1.size(), deck2.size());

        for (int i = 0; i < max_len; ++i) {
            if (i < (int)deck1.size() && (i >= (int)deck2.size() || dist(rng_) < config.crossover_rate)) {
                child_deck.push_back(deck1[i]);
            } else if (i < (int)deck2.size()) {
                child_deck.push_back(deck2[i]);
            }
        }

        // Combine pool for constraint enforcement (treat parent decks as "available cards")
        std::vector<int> pool = deck1;
        pool.insert(pool.end(), deck2.begin(), deck2.end());

        enforce_constraints(child_deck, pool);

        return child_deck;
    }

    void DeckEvolution::enforce_constraints(std::vector<int>& deck, const std::vector<int>& candidate_pool) {
         // 1. Remove excess copies (>4)
        std::map<int, int> counts;
        std::vector<int> valid_deck;

        for (int id : deck) {
            if (counts[id] < 4) {
                valid_deck.push_back(id);
                counts[id]++;
            }
        }
        deck = valid_deck;

        // 2. Adjust size to target (trim or fill)
        // Assume default target 40
        int target = 40;

        if (deck.size() > (size_t)target) {
            deck.resize(target);
        } else if (deck.size() < (size_t)target) {
            // Fill with cards from pool or duplicate existing ones
             std::uniform_int_distribution<int> pool_dist(0, candidate_pool.size() - 1);
             std::uniform_int_distribution<int> existing_dist(0, deck.size() - 1);

             while (deck.size() < (size_t)target) {
                 if (!candidate_pool.empty()) {
                     int id = candidate_pool[pool_dist(rng_)];
                     if (counts[id] < 4) {
                         deck.push_back(id);
                         counts[id]++;
                     }
                 } else if (!deck.empty()) {
                     // Try to duplicate existing
                     int id = deck[existing_dist(rng_)];
                     if (counts[id] < 4) {
                         deck.push_back(id);
                         counts[id]++;
                     } else {
                        // Edge case: all cards maxed out and no pool? break to avoid infinite loop
                        // Or just fill with anything if loose constraint.
                        // For now break if stuck
                        bool found = false;
                        for(auto const& [key, val] : counts) {
                            if (val < 4) {
                                deck.push_back(key);
                                counts[key]++;
                                found = true;
                                break;
                            }
                        }
                        if (!found) break;
                     }
                 } else {
                     break;
                 }
             }
        }
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
        if (c1.keywords.evolution) {
            for (const auto& r2 : c2.races) {
                 for (const auto& r1 : c1.races) {
                     if (r1 == r2) {
                         score += 2.0f;
                     }
                 }
            }
        }

        return score;
    }

    float DeckEvolution::calculate_interaction_score(const std::vector<int>& deck_ids) {
        if (deck_ids.empty()) return 0.0f;

        float total_synergy = 0.0f;
        int pair_count = 0;

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
