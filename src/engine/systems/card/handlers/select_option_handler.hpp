#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "core/types.hpp"
#include "core/action.hpp"

namespace dm::engine {

    class SelectOptionHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            dm::core::GameState& game_state = ctx.game_state;
            const dm::core::ActionDef& action_def = ctx.action;
            int source_instance_id = ctx.source_instance_id;
            // Controller
            dm::core::PlayerID controller = game_state.card_owner_map.size() > (size_t)source_instance_id ? game_state.card_owner_map[source_instance_id] : game_state.active_player_id;

            // This is the initial trigger of SELECT_OPTION
            // We need to queue a pending effect that asks the player to choose.

            // Validate options exist
            if (action_def.options.empty()) {
                // If no options, do nothing
                return;
            }

            // Create PendingEffect
            dm::core::PendingEffect pending(dm::core::EffectType::SELECT_OPTION, source_instance_id, controller);
            pending.options = action_def.options;
            pending.execution_context = ctx.execution_vars;

            // Add to queue
            game_state.pending_effects.push_back(pending);
        }

        void resolve_with_targets(const ResolutionContext& /*ctx*/) override {
            // Not used directly, as SELECT_OPTION resolution happens in EffectResolver via ActionType::SELECT_OPTION
        }
    };

}
