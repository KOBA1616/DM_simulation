#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"

namespace dm::engine::systems {

    class PhaseSystem {
    public:
        static PhaseSystem& instance() {
            static PhaseSystem instance;
            return instance;
        }

        void start_game(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void next_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_pass(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Phase specific logic
        void on_start_turn(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_draw_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_mana_phase(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void on_end_turn(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

    private:
        PhaseSystem() = default;
    };

}
