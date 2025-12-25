#include "meta_environment.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dm::ai {

    MetaEnvironment::MetaEnvironment(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(card_db) {
        evolver_ = std::make_unique<DeckEvolution>(card_db_);
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    void MetaEnvironment::initialize_population(const std::vector<std::vector<int>>& initial_decks) {
        population_.clear();
        for (const auto& deck : initial_decks) {
            DeckAgent agent;
            agent.id = next_agent_id_++;
            agent.deck = deck;
            agent.elo_rating = 1500.0f;
            agent.generation = 0;
            population_.push_back(agent);
        }

        // Fill to population size if needed (with copies or mutations)
        while (population_.size() < (size_t)config_.population_size && !population_.empty()) {
             DeckAgent parent = population_[population_.size() % initial_decks.size()];
             DeckAgent child = parent;
             child.id = next_agent_id_++;
             // child.deck = evolver_->evolve_deck(parent.deck, {}); // Mutate slightly?
             population_.push_back(child);
        }
    }

    void MetaEnvironment::record_match(int agent1_idx, int agent2_idx, int winner_idx) {
        if (agent1_idx < 0 || agent1_idx >= (int)population_.size()) return;
        if (agent2_idx < 0 || agent2_idx >= (int)population_.size()) return;

        DeckAgent& p1 = population_[agent1_idx];
        DeckAgent& p2 = population_[agent2_idx];

        p1.matches_played++;
        p2.matches_played++;

        float score1 = 0.5f;
        float score2 = 0.5f;

        if (winner_idx == 1) { // Player 1 won
            score1 = 1.0f;
            score2 = 0.0f;
            p1.wins++;
        } else if (winner_idx == 2) { // Player 2 won
            score1 = 0.0f;
            score2 = 1.0f;
            p2.wins++;
        }

        // Elo update K-factor
        float k = 32.0f;

        float expected1 = calculate_expected_score(p1.elo_rating, p2.elo_rating);
        float expected2 = calculate_expected_score(p2.elo_rating, p1.elo_rating);

        p1.elo_rating += k * (score1 - expected1);
        p2.elo_rating += k * (score2 - expected2);
    }

    float MetaEnvironment::calculate_expected_score(float elo_a, float elo_b) {
        return 1.0f / (1.0f + std::pow(10.0f, (elo_b - elo_a) / 400.0f));
    }

    void MetaEnvironment::sort_population() {
        std::sort(population_.begin(), population_.end(), [](const DeckAgent& a, const DeckAgent& b) {
            return a.elo_rating > b.elo_rating;
        });
    }

    DeckAgent MetaEnvironment::get_agent(int index) const {
        if (index >= 0 && index < (int)population_.size()) {
            return population_[index];
        }
        return DeckAgent();
    }

    void MetaEnvironment::step_generation(const std::vector<int>& candidate_pool) {
        sort_population();

        int elite_cutoff = config_.elite_count;
        int replace_start = std::max(elite_cutoff, (int)population_.size() - config_.replacement_count);

        std::vector<DeckAgent> next_gen;

        // 1. Keep Elites
        for (int i = 0; i < elite_cutoff && i < (int)population_.size(); ++i) {
            population_[i].generation++; // Survived
            next_gen.push_back(population_[i]);
        }

        // 2. Fill the rest with offspring
        std::uniform_int_distribution<int> parent_dist(0, replace_start - 1); // Select from top half roughly

        while (next_gen.size() < population_.size()) {
            // Select parents (Tournament or Rank based? Simple random from top tier for now)
            int p1_idx = parent_dist(rng_);
            int p2_idx = parent_dist(rng_);

            const auto& parent1 = population_[p1_idx];
            const auto& parent2 = population_[p2_idx];

            DeckAgent child;
            child.id = next_agent_id_++;
            child.generation = std::max(parent1.generation, parent2.generation) + 1;

            // Crossover
            child.deck = evolver_->crossover_decks(parent1.deck, parent2.deck, config_.evo_config);

            // Mutation
            child.deck = evolver_->evolve_deck(child.deck, candidate_pool, config_.evo_config);

            child.elo_rating = (parent1.elo_rating + parent2.elo_rating) / 2.0f; // Start with avg elo? or reset?
            // Resetting to 1500 might be unfair if population average drifts.
            // Averaging parent Elo is a decent heuristic for "expected strength".

            next_gen.push_back(child);
        }

        population_ = next_gen;
    }

}
