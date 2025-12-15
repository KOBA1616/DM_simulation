#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "core/game_state.hpp"
#include "core/action.hpp" // Corrected include
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class BlockHandler {
    public:
        static void resolve(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            // 1. Set Blocked State
            // int_value = blocker_instance_id
            MutateCommand block_cmd(-1, MutateCommand::MutationType::SET_BLOCKER, action.source_instance_id);
            block_cmd.execute(game_state);

            // 2. Tap Blocker
            Player& defender = game_state.get_non_active_player();
            auto it = std::find_if(defender.battle_zone.begin(), defender.battle_zone.end(),
                 [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });

            if (it != defender.battle_zone.end()) {
                 MutateCommand tap_cmd(action.source_instance_id, MutateCommand::MutationType::TAP);
                 tap_cmd.execute(game_state);

                 // 3. Trigger ON_BLOCK
                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_BLOCK, action.source_instance_id, card_db);
            }

            // 4. Queue Battle Resolution
            game_state.pending_effects.emplace_back(EffectType::RESOLVE_BATTLE, action.source_instance_id, game_state.active_player_id);
        }
    };
}
