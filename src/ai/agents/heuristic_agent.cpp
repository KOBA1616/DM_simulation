#include "heuristic_agent.hpp"
#include <algorithm>
#include <iostream>

namespace dm::ai {

    HeuristicAgent::HeuristicAgent(int player_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : player_id_(player_id), card_db_(card_db) {
        // Initialize RNG with a random seed
        std::random_device rd;
        rng_ = std::mt19937(rd());
    }

    const dm::core::CardDefinition* HeuristicAgent::get_def(dm::core::CardID cid) const {
        auto it = card_db_.find(cid);
        if (it != card_db_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    dm::core::Action HeuristicAgent::get_action(const dm::core::GameState& state,
                                                const std::vector<dm::core::Action>& legal_actions) {
        if (legal_actions.empty()) {
            return dm::core::Action(); // Should not happen if game is not over
        }

        using namespace dm::core;

        // 1. Mana Charge
        std::vector<Action> mana_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::MANA_CHARGE || action.type == ActionType::MOVE_CARD) {
                mana_actions.push_back(action);
            }
        }

        if (!mana_actions.empty()) {
            int current_mana = state.players[player_id_].mana_zone.size();
            if (current_mana < 8) {
                // Return random mana charge
                std::uniform_int_distribution<> dist(0, mana_actions.size() - 1);
                return mana_actions[dist(rng_)];
            }
        }

        // 2. Play Card
        std::vector<Action> play_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::PLAY_CARD) {
                play_actions.push_back(action);
            }
        }

        if (!play_actions.empty()) {
            // Play the most expensive card possible
            std::sort(play_actions.begin(), play_actions.end(), [&](const Action& a, const Action& b) {
                const auto* def_a = get_def(a.card_id);
                const auto* def_b = get_def(b.card_id);
                int cost_a = def_a ? def_a->cost : 0;
                int cost_b = def_b ? def_b->cost : 0;
                return cost_a > cost_b; // Descending
            });
            return play_actions[0];
        }

        // 3. Attack Player (Aggro)
        std::vector<Action> attack_player_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::ATTACK_PLAYER) {
                attack_player_actions.push_back(action);
            }
        }

        if (!attack_player_actions.empty()) {
            std::uniform_int_distribution<> dist(0, attack_player_actions.size() - 1);
            return attack_player_actions[dist(rng_)];
        }

        // 4. Attack Creature
        std::vector<Action> attack_creature_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::ATTACK_CREATURE) {
                attack_creature_actions.push_back(action);
            }
        }

        if (!attack_creature_actions.empty()) {
            std::uniform_int_distribution<> dist(0, attack_creature_actions.size() - 1);
            return attack_creature_actions[dist(rng_)];
        }

        // 5. Block
        std::vector<Action> block_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::BLOCK) {
                block_actions.push_back(action);
            }
        }

        if (!block_actions.empty()) {
            // Block if shields are low or 50% chance
            int shield_count = state.players[player_id_].shield_zone.size();
            bool should_block = false;
            if (shield_count <= 2) {
                should_block = true;
            } else {
                std::uniform_real_distribution<> dist(0.0, 1.0);
                if (dist(rng_) < 0.5) {
                    should_block = true;
                }
            }

            if (should_block) {
                std::uniform_int_distribution<> dist(0, block_actions.size() - 1);
                return block_actions[dist(rng_)];
            }
        }

        // 6. Select Target
        std::vector<Action> select_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::SELECT_TARGET) {
                select_actions.push_back(action);
            }
        }

        if (!select_actions.empty()) {
            std::uniform_int_distribution<> dist(0, select_actions.size() - 1);
            return select_actions[dist(rng_)];
        }

        // 7. Shield Trigger
        std::vector<Action> st_actions;
        for (const auto& action : legal_actions) {
            if (action.type == ActionType::USE_SHIELD_TRIGGER) {
                st_actions.push_back(action);
            }
        }

        if (!st_actions.empty()) {
            // Always use first trigger for now
            return st_actions[0];
        }

        // Default: Random choice from all legal actions
        std::uniform_int_distribution<> dist(0, legal_actions.size() - 1);
        return legal_actions[dist(rng_)];
    }

}
