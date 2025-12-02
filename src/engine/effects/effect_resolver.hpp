#pragma once
#include "../../core/game_state.hpp"
#include "../../core/action.hpp"
#include "../../core/card_def.hpp"
#include <map>

namespace dm::engine {

    class EffectResolver {
    public:
        static void resolve_action(
            dm::core::GameState& game_state, 
            const dm::core::Action& action,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db
        );

    private:
        static void resolve_play_card(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void resolve_mana_charge(dm::core::GameState& game_state, const dm::core::Action& action);
        static void resolve_attack(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void resolve_pending_effect(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void resolve_use_shield_trigger(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void resolve_block(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void resolve_select_target(dm::core::GameState& game_state, const dm::core::Action& action);
        static void execute_battle(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Helper functions for generic keyword handling
        static int get_creature_power(const dm::core::CardInstance& creature, const dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static int get_breaker_count(const dm::core::CardInstance& creature, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
    };

}
