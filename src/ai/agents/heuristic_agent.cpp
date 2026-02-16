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

    dm::core::CardID HeuristicAgent::get_card_id(const dm::core::GameState& state, int instance_id) const {
        // Safe access (const method assumed on GameState)
        // If get_card_instance is not const, we might need const_cast or rely on API
        // Assuming const auto* get_card_instance(int) const exists.
        // If not, we might have issues. Let's assume standard const access.
        // Actually, state.players is public, we can scan. But get_card_instance is better.
        // Based on other files, it seems to return pointer.
        // The implementation in GameState likely supports const.
        // If compilation fails, we fix it.
        const auto* ptr = state.get_card_instance(instance_id);
        if (ptr) return ptr->card_id;
        return 0;
    }

    dm::core::CommandDef HeuristicAgent::get_command(const dm::core::GameState& state,
                                                const std::vector<dm::core::CommandDef>& legal_actions) {
        if (legal_actions.empty()) {
            return dm::core::CommandDef(); // Should not happen if game is not over
        }

        using namespace dm::core;

        // 1. Mana Charge
        std::vector<CommandDef> mana_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::MANA_CHARGE ||
               (action.type == CommandType::MOVE_CARD && action.to_zone == "MANA_ZONE")) {
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
        std::vector<CommandDef> play_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::PLAY_FROM_ZONE ||
                action.type == CommandType::CAST_SPELL ||
                action.type == CommandType::SUMMON_TOKEN) {
                play_actions.push_back(action);
            }
        }

        if (!play_actions.empty()) {
            // Play the most expensive card possible
            std::sort(play_actions.begin(), play_actions.end(), [&](const CommandDef& a, const CommandDef& b) {
                CardID cid_a = get_card_id(state, a.instance_id);
                CardID cid_b = get_card_id(state, b.instance_id);
                const auto* def_a = get_def(cid_a);
                const auto* def_b = get_def(cid_b);
                int cost_a = def_a ? def_a->cost : 0;
                int cost_b = def_b ? def_b->cost : 0;
                return cost_a > cost_b; // Descending
            });
            return play_actions[0];
        }

        // 3. Attack Player (Aggro)
        std::vector<CommandDef> attack_player_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::ATTACK_PLAYER) {
                attack_player_actions.push_back(action);
            }
        }

        if (!attack_player_actions.empty()) {
            std::uniform_int_distribution<> dist(0, attack_player_actions.size() - 1);
            return attack_player_actions[dist(rng_)];
        }

        // 4. Attack Creature
        std::vector<CommandDef> attack_creature_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::ATTACK_CREATURE) {
                attack_creature_actions.push_back(action);
            }
        }

        if (!attack_creature_actions.empty()) {
            std::uniform_int_distribution<> dist(0, attack_creature_actions.size() - 1);
            return attack_creature_actions[dist(rng_)];
        }

        // 5. Block
        std::vector<CommandDef> block_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::BLOCK) {
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
        std::vector<CommandDef> select_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::SELECT_TARGET) {
                select_actions.push_back(action);
            }
        }

        if (!select_actions.empty()) {
            std::uniform_int_distribution<> dist(0, select_actions.size() - 1);
            return select_actions[dist(rng_)];
        }

        // 7. Shield Trigger
        std::vector<CommandDef> st_actions;
        for (const auto& action : legal_actions) {
            if (action.type == CommandType::SHIELD_TRIGGER) {
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
