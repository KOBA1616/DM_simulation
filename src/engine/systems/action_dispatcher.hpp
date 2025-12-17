#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include <map>

namespace dm::engine::systems {

    class ActionDispatcher {
    public:
        // Central dispatcher for all actions
        static void dispatch(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        static void handle_select_target(core::GameState& game_state, const core::Action& action);
        static void handle_select_option(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_select_number(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_use_ability(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_declare_reaction(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
        static void handle_resolve_effect(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db);
    };

}
