#pragma once

#include "deck_evolution.hpp"
#include <vector>
#include <map>
#include <memory>
#include <string>

namespace dm::ai {

    struct DeckAgent {
        int id;
        std::vector<int> deck;
        float elo_rating = 1500.0f;
        int matches_played = 0;
        int wins = 0;
        int generation = 0;
        std::string archetype; // Optional label
    };

    struct MetaConfig {
        int population_size = 20;
        int elite_count = 4; // Top N to keep
        int replacement_count = 4; // Bottom N to replace
        DeckEvolutionConfig evo_config;
    };

    class MetaEnvironment {
    public:
        MetaEnvironment(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Initialize population (e.g. from random or initial list)
        void initialize_population(const std::vector<std::vector<int>>& initial_decks);

        // Register a match result to update Elo
        void record_match(int agent1_idx, int agent2_idx, int winner_idx); // winner_idx = 1 or 2, 0 for draw

        // Advance generation: Selection -> Crossover -> Mutation
        void step_generation(const std::vector<int>& candidate_pool);

        // Getters
        const std::vector<DeckAgent>& get_population() const { return population_; }
        DeckAgent get_agent(int index) const;

        // Export/Import
        // void save_to_file(const std::string& path);
        // void load_from_file(const std::string& path);

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        std::vector<DeckAgent> population_;
        std::unique_ptr<DeckEvolution> evolver_;
        MetaConfig config_;
        int next_agent_id_ = 0;
        std::mt19937 rng_;

        void sort_population(); // Sort by Elo
        float calculate_expected_score(float elo_a, float elo_b);
    };

}
