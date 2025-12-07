#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../target_utils.hpp" // For helper logic if needed, though get_controller is in GenericCardSystem

namespace dm::engine {

    // Helper to get controller (duplicated from GenericCardSystem temporarily until shared util is established)
    // Actually, let's use TargetUtils or just implement it inline since it's simple.
    // Or better: Use the one in GenericCardSystem if we can expose it.
    // Ideally, GameState should have get_card_controller(instance_id).

    class DrawHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
             using namespace dm::core;

             // We need controller.
             PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
             Player& controller = game_state.players[controller_id];

             // Handle Variable Linking
             int count = action.value1;
             if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
                count = execution_context[action.input_value_key];
             }
             if (count == 0 && !action.value.empty()) {
                 try { count = std::stoi(action.value); } catch (...) {}
             }
             if (count == 0) count = 1;

             // Execute Draw
             for (int i = 0; i < count; ++i) {
                if (controller.deck.empty()) {
                    game_state.winner = (controller.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                    return;
                }
                CardInstance c = controller.deck.back();
                controller.deck.pop_back();
                controller.hand.push_back(c);

                if (controller.id == game_state.active_player_id) {
                    game_state.turn_stats.cards_drawn_this_turn++;
                }
             }
        }
    };
}
