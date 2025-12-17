#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "core/card_def.hpp"
#include <map>

namespace dm::engine::systems {

    class PlaySystem {
    public:
        // Handle ActionType::PLAY_CARD / DECLARE_PLAY
        static void handle_play_card(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Handle ActionType::PAY_COST
        static void handle_pay_cost(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Handle ActionType::RESOLVE_PLAY (Move from Stack to Zone)
        static void resolve_play_from_stack(core::GameState& game_state, int stack_instance_id, int cost_reduction, core::SpawnSource spawn_source, core::PlayerID controller, const std::map<core::CardID, core::CardDefinition>& card_db, int evo_source_id = -1, int dest_override = 0);

        // Handle ActionType::MANA_CHARGE
        static void handle_mana_charge(core::GameState& game_state, const core::Action& action);
    };

}
