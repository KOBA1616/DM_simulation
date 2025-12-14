#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <algorithm>

namespace dm::engine {

    class TokenHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            spawn_token(ctx);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             spawn_token(ctx);
        }

    private:
        void spawn_token(const ResolutionContext& ctx) {
            using namespace dm::core;

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            Player& controller = ctx.game_state.players[controller_id];

            // Determine Token ID or Name
            // value1 might be ID, str_val might be name
            // CardRegistry lookup

            int token_id = ctx.action.value1;
            // If ID is 0, maybe lookup by name?
            // Currently registry is by ID.

            // NOTE: Token IDs should be defined in JSON or known.
            // If token_id is provided, use it.

            if (token_id <= 0) return;

            // Check if ID exists in Registry (Tokens are usually in Registry)
            const auto& registry = CardRegistry::get_all_definitions();
            if (registry.count(token_id)) {
                 // Create Instance
                 // We need a unique instance ID.
                 // GameState usually tracks max instance ID?
                 // Currently GameInstance manages it, but GameState passed here doesn't have "get_next_instance_id".
                 // In `GenericCardSystem`, we don't handle ID generation explicitly?
                 // Wait, `GenericCardSystem` usually works on existing instances.
                 // Creating new instances requires ID generation.

                 // How does `GameInstance` handle it?
                 // `GameInstance` has `next_instance_id`.
                 // `GameState` does not.

                 // This is a problem for pure `GameState` manipulation without `GameInstance`.
                 // However, we can find the max instance ID in the state and increment.
                 // It's O(N) but safe.

                 int max_id = 0;
                 // Check owners map size?
                 max_id = (int)ctx.game_state.card_owner_map.size();
                 // Instance IDs are 0 to size-1?
                 // card_owner_map is vector. index = instance_id.
                 // So next ID is card_owner_map.size().

                 int new_inst_id = max_id;

                 CardInstance token_inst(token_id, new_inst_id);
                 token_inst.turn_played = ctx.game_state.turn_number;
                 token_inst.summoning_sickness = true; // Tokens are creatures usually

                 // Add to Owner Map
                 ctx.game_state.card_owner_map.push_back(controller_id);

                 // Add to Battle Zone
                 controller.battle_zone.push_back(token_inst);

                 // Trigger ON_PLAY
                 GenericCardSystem::resolve_trigger(ctx.game_state, TriggerType::ON_PLAY, new_inst_id, ctx.card_db);
            }
        }
    };
}
