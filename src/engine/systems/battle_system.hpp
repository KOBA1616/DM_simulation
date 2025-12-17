#pragma once
#include "core/game_state.hpp"
#include "core/game_event.hpp"
#include "core/action.hpp"
#include <map>

namespace dm::engine::systems {

    class BattleSystem {
    public:
        // Handle ActionType::ATTACK_PLAYER and ATTACK_CREATURE
        static void handle_attack(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Handle ActionType::BLOCK
        static void handle_block(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Handle ActionType::RESOLVE_BATTLE (Creature vs Creature)
        static void resolve_battle(core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Handle ActionType::BREAK_SHIELD
        static void resolve_break_shield(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        // Helper to determine power
        static int get_creature_power(const core::CardInstance& creature, const core::GameState& game_state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Helper to determine breaker capability
        static int get_breaker_count(const core::CardInstance& creature, const std::map<core::CardID, core::CardDefinition>& card_db);
    };

}
