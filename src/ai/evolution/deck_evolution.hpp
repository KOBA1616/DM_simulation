#pragma once

#include "core/types.hpp"
#include "core/card_def.hpp"
#include "core/card_stats.hpp"
#include <vector>
#include <map>
#include <string>
#include <random>

namespace dm::ai {

    struct DeckEvolutionConfig {
        int target_deck_size = 40;
        float mutation_rate = 0.1f; // Probability of changing a card
        float crossover_rate = 0.5f; // Fraction of genes from parent 1 (or mix ratio)
        float synergy_weight = 1.0f;
        float curve_weight = 0.5f;
    };

    class DeckEvolution {
    public:
        DeckEvolution(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Main evolution step: takes a deck and returns a mutated version
        std::vector<int> evolve_deck(const std::vector<int>& current_deck,
                                     const std::vector<int>& candidate_pool,
                                     const DeckEvolutionConfig& config = DeckEvolutionConfig());

        // Crossover step: combines two decks into one
        std::vector<int> crossover_decks(const std::vector<int>& deck1,
                                         const std::vector<int>& deck2,
                                         const DeckEvolutionConfig& config = DeckEvolutionConfig());

        // Calculate interaction/synergy score for a deck
        float calculate_interaction_score(const std::vector<int>& deck_ids);

        // Helper to get candidates (e.g. filtered by civ)
        std::vector<int> get_candidates_by_civ(const std::vector<int>& pool, dm::core::Civilization civ);

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        std::mt19937 rng_;

        // Helper to check synergy between two cards
        float get_pair_synergy(const dm::core::CardDefinition& card1, const dm::core::CardDefinition& card2);

        // Helper to calculate mana curve score (deviation from ideal)
        float calculate_curve_score(const std::vector<int>& deck_ids);

        // Helper to enforce deck construction rules
        void enforce_constraints(std::vector<int>& deck, const std::vector<int>& candidate_pool);
    };

}
