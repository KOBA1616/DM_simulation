#include "pimc_generator.hpp"
#include <algorithm>
#include <iostream>
#include <set>

namespace dm::ai::inference {

    PimcGenerator::PimcGenerator(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db)
        : card_db_(card_db) {
    }

    void PimcGenerator::set_inference_model(std::shared_ptr<DeckInference> inference) {
        inference_ = inference;
    }

    dm::core::GameState PimcGenerator::generate_determinized_state(
        const dm::core::GameState& observed_state,
        dm::core::PlayerID observer_id,
        uint32_t seed
    ) {
        // GameState is not copyable, so we must clone it.
        dm::core::GameState determinized = observed_state.clone();
        dm::core::PlayerID opponent_id = 1 - observer_id;
        auto& opp = determinized.players[opponent_id];

        // 3. Get sample pool
        std::vector<dm::core::CardID> hidden_pool;
        if (inference_) {
             hidden_pool = inference_->sample_hidden_cards(observed_state, observer_id, seed);
        } else {
             // Fallback: Fill with dummy cards (ID 1?) if no inference model
             int slots_needed = 0;
             for (const auto& c : opp.hand) if (c.card_id == 0) slots_needed++;
             for (const auto& c : opp.deck) if (c.card_id == 0) slots_needed++;
             for (const auto& c : opp.shield_zone) if (c.card_id == 0) slots_needed++;

             hidden_pool.resize(slots_needed, 1);
        }

        // 4. Shuffle the pool
        std::mt19937 rng(seed);
        std::shuffle(hidden_pool.begin(), hidden_pool.end(), rng);

        // 5. Fill Hidden Slots
        size_t pool_idx = 0;

        // Helper lambda to fill zone
        auto fill_zone = [&](std::vector<dm::core::CardInstance>& zone, bool check_face_down = false) {
            for (auto& inst : zone) {
                bool is_hidden = (inst.card_id == 0);

                if (is_hidden) {
                    if (pool_idx < hidden_pool.size()) {
                        inst.card_id = hidden_pool[pool_idx++];
                        if (card_db_ && card_db_->count(inst.card_id)) {
                             // inst.power_modifier = 0; // Reset mods? No, keep state.
                        }
                    } else {
                        // Pool exhausted
                        inst.card_id = 1;
                    }
                }
            }
        };

        fill_zone(opp.hand);
        fill_zone(opp.deck);
        fill_zone(opp.shield_zone);

        return determinized;
    }

    dm::core::GameState PimcGenerator::generate_determinized_state(
        const dm::core::GameState& observed_state,
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
        dm::core::PlayerID observer_id,
        const std::vector<dm::core::CardID>& opponent_deck_candidates,
        uint32_t seed
    ) {
        dm::core::GameState determinized = observed_state.clone();
        dm::core::PlayerID opponent_id = 1 - observer_id;
        auto& opp = determinized.players[opponent_id];

        std::vector<dm::core::CardID> hidden_pool = opponent_deck_candidates;

        std::mt19937 rng(seed);
        std::shuffle(hidden_pool.begin(), hidden_pool.end(), rng);

        size_t pool_idx = 0;

        auto fill_zone = [&](std::vector<dm::core::CardInstance>& zone) {
            for (auto& inst : zone) {
                if (inst.card_id == 0) {
                    if (pool_idx < hidden_pool.size()) {
                        inst.card_id = hidden_pool[pool_idx++];
                    } else {
                        inst.card_id = 1; // Fallback
                    }
                }
            }
        };

        fill_zone(opp.hand);
        fill_zone(opp.deck);
        fill_zone(opp.shield_zone);

        return determinized;
    }

}
