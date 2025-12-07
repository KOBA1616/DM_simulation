#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
             using namespace dm::core;

             PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
             Player& controller = game_state.players[controller_id];

             int count = action.value1;
             if (!action.input_value_key.empty() && execution_context.count(action.input_value_key)) {
                count = execution_context[action.input_value_key];
             }
             if (count == 0) count = 1;

             for (int i = 0; i < count; ++i) {
                if (controller.deck.empty()) break;
                CardInstance c = controller.deck.back();
                controller.deck.pop_back();
                c.is_tapped = false;
                controller.mana_zone.push_back(c);
             }
        }
    };
}
