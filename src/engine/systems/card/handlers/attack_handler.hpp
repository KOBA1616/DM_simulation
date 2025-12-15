#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/flow/reaction_system.hpp"

namespace dm::engine {

    class AttackHandler {
    public:
        static void handle_attack(dm::core::GameState& game_state, const dm::core::Action& action, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;

            // 1. Validate Attacker
            int attacker_id = action.source_instance_id;
            Player& attacker = game_state.get_active_player();

            auto it = std::find_if(attacker.battle_zone.begin(), attacker.battle_zone.end(),
                [&](const CardInstance& c){ return c.instance_id == attacker_id; });

            if (it == attacker.battle_zone.end()) {
                // Attacker not found in battle zone
                return;
            }

            // 2. Tap Attacker (MutateCommand)
            if (!it->is_tapped) {
                 game_command::MutateCommand cmd(attacker_id, game_command::MutateCommand::MutationType::TAP);
                 cmd.execute(game_state);
            }

            // 3. Update Game State (Attack Info)
            int target_id = -1;
            int target_player = -1;

            if (action.type == ActionType::ATTACK_CREATURE) {
                target_id = action.target_instance_id;
            } else if (action.type == ActionType::ATTACK_PLAYER) {
                target_player = action.target_player;
            }

            // Using FlowCommand for state updates
            game_command::FlowCommand cmd_source(game_command::FlowCommand::FlowType::SET_ATTACK_SOURCE, attacker_id);
            cmd_source.execute(game_state);

            game_command::FlowCommand cmd_target_id(game_command::FlowCommand::FlowType::SET_ATTACK_TARGET, target_id);
            cmd_target_id.execute(game_state);

            game_command::FlowCommand cmd_target_player(game_command::FlowCommand::FlowType::SET_ATTACK_PLAYER, target_player);
            cmd_target_player.execute(game_state);

            // Increment stats
            game_state.turn_stats.attacks_declared_this_turn++;

            // 4. Resolve Triggers (ON_ATTACK)
            GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_ATTACK, attacker_id, card_db);

            // 5. Change Phase (FlowCommand)
            if (game_state.current_phase == Phase::ATTACK) {
                 game_command::FlowCommand cmd_phase(game_command::FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::BLOCK));
                 cmd_phase.execute(game_state);
            }

            // 6. Reaction Window
            PlayerID defender_id = game_state.get_non_active_player().id;
            ReactionSystem::check_and_open_window(game_state, card_db, "ON_ATTACK", defender_id);
        }
    };
}
