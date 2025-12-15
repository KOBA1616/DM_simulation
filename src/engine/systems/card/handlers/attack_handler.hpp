#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/flow/reaction_system.hpp"
#include "core/game_state.hpp"
#include "core/action.hpp" // Corrected include
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class AttackHandler {
    public:
        static void resolve(core::GameState& game_state, const core::Action& action, const std::map<core::CardID, core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            // 1. Identify Attacker
            Player& attacker_player = game_state.get_active_player();
            auto it = std::find_if(attacker_player.battle_zone.begin(), attacker_player.battle_zone.end(),
                [&](const CardInstance& c){ return c.instance_id == action.source_instance_id; });

            if (it == attacker_player.battle_zone.end()) return;

            // 2. Execute GameCommands for State Changes

            // Tap the attacker
            MutateCommand tap_cmd(action.source_instance_id, MutateCommand::MutationType::TAP);
            tap_cmd.execute(game_state);

            // Set Source
            MutateCommand source_cmd(-1, MutateCommand::MutationType::SET_ATTACK_SOURCE, action.source_instance_id);
            source_cmd.execute(game_state);

            // Set Target
            if (action.type == ActionType::ATTACK_CREATURE) {
                MutateCommand target_cmd(-1, MutateCommand::MutationType::SET_ATTACK_TARGET, action.target_instance_id);
                target_cmd.execute(game_state);
                // Clear Player Target
                MutateCommand clear_player(-1, MutateCommand::MutationType::SET_ATTACK_PLAYER, 255);
                clear_player.execute(game_state);
            } else {
                MutateCommand target_cmd(-1, MutateCommand::MutationType::SET_ATTACK_PLAYER, (int)action.target_player);
                target_cmd.execute(game_state);
                 // Clear Creature Target
                MutateCommand clear_creature(-1, MutateCommand::MutationType::SET_ATTACK_TARGET, -1);
                clear_creature.execute(game_state);
            }

            // Reset Blocker
            MutateCommand reset_blocker(-1, MutateCommand::MutationType::SET_BLOCKER, -1);
            reset_blocker.execute(game_state);

            // 3. Stats
            game_state.turn_stats.attacks_declared_this_turn++;

            // 4. Triggers
            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_ATTACK, action.source_instance_id, card_db);

            // 5. Phase Transition
            if (game_state.current_phase == Phase::ATTACK) {
                FlowCommand phase_cmd(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::BLOCK));
                phase_cmd.execute(game_state);
            }

            // 6. Reaction Window
            Player& defender = game_state.get_non_active_player();
            ReactionSystem::check_and_open_window(game_state, card_db, "ON_ATTACK", defender.id);
        }
    };
}
