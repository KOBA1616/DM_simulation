#include "action_commands.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "commands.hpp" // For TransitionCommand
#include <fstream>  // For debug logging

namespace dm::engine::game_command {

    using namespace dm::engine::systems;

    void PlayCardCommand::execute(core::GameState& state) {
        // Construct CommandDef to pass to GameLogicSystem
        const auto& card_db = CardRegistry::get_all_definitions();

        core::CommandDef cmd;
        cmd.type = core::CommandType::PLAY_FROM_ZONE;
        cmd.instance_id = card_instance_id;
        // Use amount to signal spell side (1 = spell side, 0 = creature side)
        cmd.amount = is_spell_side ? 1 : 0;

        // Note: spawn_source is currently handled by inference in dispatch_command via get_card_location,
        // or could be added to CommandDef if explicitly needed in future.

        GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
    }

    void PlayCardCommand::invert(core::GameState& state) {
        // No-op
        (void)state;
    }

    void AttackCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::CommandDef cmd;
        if (target_id == -1) {
            cmd.type = core::CommandType::ATTACK_PLAYER;
            // target_instance = -1 implies attacking the opponent player (standard 1v1)
            cmd.target_instance = -1;
        } else {
            cmd.type = core::CommandType::ATTACK_CREATURE;
            cmd.target_instance = target_id;
        }
        cmd.instance_id = source_id;

        GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
    }

    void AttackCommand::invert(core::GameState& state) {
        (void)state;
    }

    void BlockCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::CommandDef cmd;
        cmd.type = core::CommandType::BLOCK;
        cmd.instance_id = blocker_id;

        GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
    }

    void BlockCommand::invert(core::GameState& state) {
        (void)state;
    }

    void UseAbilityCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();

        core::CommandDef cmd;
        cmd.type = core::CommandType::USE_ABILITY;
        cmd.instance_id = source_id;
        cmd.target_instance = target_id;

        GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
    }

    void UseAbilityCommand::invert(core::GameState& state) {
        (void)state;
    }

    void ManaChargeCommand::execute(core::GameState& state) {
        using namespace dm::core;
        
        try {
            std::ofstream lout("logs/manacharge_trace.txt", std::ios::app);
            if (lout) {
                lout << "MANA_CHARGE_CMD CALLED id=" << card_id << "\n";
                lout.close();
            }
        } catch(...) {}
        
        const CardInstance* card_ptr = state.get_card_instance(card_id);
        if (!card_ptr) {
            return;
        }

        PlayerID owner = card_ptr->owner;
        
        if (state.turn_stats.mana_charged_by_player[owner]) {
            return;
        }
        
        bool found = false;
        const Player& p = state.players[owner];
        for(const auto& c : p.hand) {
            if(c.instance_id == card_id) {
                found = true;
                break;
            }
        }
        
        if (!found) {
            return;
        }

        auto move_cmd = std::make_shared<TransitionCommand>(card_id, Zone::HAND, Zone::MANA, owner);
        state.execute_command(std::move(move_cmd));
        
        auto flow_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_MANA_CHARGED, 1);
        state.execute_command(std::move(flow_cmd));
    }

    void ManaChargeCommand::invert(core::GameState& state) {
        (void)state;
    }

    void PassCommand::execute(core::GameState& state) {
        const auto& card_db = CardRegistry::get_all_definitions();
        core::CommandDef cmd;
        cmd.type = core::CommandType::PASS;

        GameLogicSystem::resolve_command_oneshot(state, cmd, card_db);
    }

    void PassCommand::invert(core::GameState& state) {
        (void)state;
    }

}
