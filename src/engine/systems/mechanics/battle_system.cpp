#include "battle_system.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/systems/rules/restriction_system.hpp"
#include "engine/systems/effects/trigger_system.hpp"
#include <iostream>
#include <fstream>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void BattleSystem::handle_attack(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                     const std::map<CardID, CardDefinition>& card_db) {
        int instance_id = exec.resolve_int(inst.args.value("source", 0));
        int target_id = exec.resolve_int(inst.args.value("target", -1));

        const CardInstance* card = state.get_card_instance(instance_id);
        if (!card || !card_db.count(card->card_id)) return;
        const auto& def = card_db.at(card->card_id);

        if (RestrictionSystem::instance().is_attack_forbidden(state, *card, def, target_id, card_db)) return;

        auto cmd_src = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_SOURCE, instance_id);
        state.execute_command(std::move(cmd_src));

        auto cmd_tgt = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_TARGET, target_id);
        state.execute_command(std::move(cmd_tgt));

        auto cmd_blk = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_BLOCKING_CREATURE, -1);
        state.execute_command(std::move(cmd_blk));

        int target_player = -1;
        if (target_id == -1) {
            target_player = 1 - state.active_player_id;
        }
        auto cmd_plyr = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ATTACK_PLAYER, target_player);
        state.execute_command(std::move(cmd_plyr));

        auto cmd_tap = std::make_shared<MutateCommand>(instance_id, MutateCommand::MutationType::TAP);
        state.execute_command(std::move(cmd_tap));

        auto phase_cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::BLOCK));
        state.execute_command(std::move(phase_cmd));
    }

    void BattleSystem::handle_block(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                    const std::map<CardID, CardDefinition>& card_db) {
        int blocker_id = exec.resolve_int(inst.args.value("blocker", -1));
        if (blocker_id == -1) return;

        const CardInstance* blocker = state.get_card_instance(blocker_id);
        if (!blocker || !card_db.count(blocker->card_id)) return;
        const auto& def = card_db.at(blocker->card_id);

        if (RestrictionSystem::instance().is_block_forbidden(state, *blocker, def, card_db)) return;

        auto cmd_tap = std::make_shared<MutateCommand>(blocker_id, MutateCommand::MutationType::TAP);
        state.execute_command(std::move(cmd_tap));

        auto cmd_blk = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_BLOCKING_CREATURE, blocker_id);
        state.execute_command(std::move(cmd_blk));

        if (blocker && card_db.count(blocker->card_id)) {
             TriggerSystem::instance().resolve_trigger(state, TriggerType::ON_BLOCK, blocker_id, card_db);
        }

        PendingEffect pe(EffectType::RESOLVE_BATTLE, state.current_attack.source_instance_id, state.active_player_id);
        pe.target_instance_ids = {blocker_id};
        pe.execution_context["attacker"] = state.current_attack.source_instance_id;
        pe.execution_context["defender"] = blocker_id;

        TriggerSystem::instance().add_pending_effect(state, pe);
    }

    void BattleSystem::handle_resolve_battle(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                             const std::map<CardID, CardDefinition>& card_db) {
        int attacker_id = exec.resolve_int(inst.args.value("attacker", -1));
        int defender_id = exec.resolve_int(inst.args.value("defender", -1));

        const CardInstance* attacker = state.get_card_instance(attacker_id);
        const CardInstance* defender = state.get_card_instance(defender_id);

        if (!attacker || !defender) return;

        int power_attacker = get_creature_power(*attacker, state, card_db);
        int power_defender = get_creature_power(*defender, state, card_db);

        bool attacker_dies = false;
        bool defender_dies = false;

        bool attacker_slayer = false;
        if (card_db.count(attacker->card_id)) attacker_slayer = card_db.at(attacker->card_id).keywords.slayer;

        bool defender_slayer = false;
        if (card_db.count(defender->card_id)) defender_slayer = card_db.at(defender->card_id).keywords.slayer;

        if (power_attacker > power_defender) {
            defender_dies = true;
            if (defender_slayer) attacker_dies = true;
        } else if (power_attacker < power_defender) {
            attacker_dies = true;
            if (attacker_slayer) defender_dies = true;
        } else {
            attacker_dies = true;
            defender_dies = true;
        }

        std::vector<Instruction> generated;

        if (attacker_dies) {
            Instruction move(InstructionOp::MOVE);
            move.args["target"] = attacker_id;
            move.args["to"] = "GRAVEYARD";
            generated.push_back(move);
        }

        if (defender_dies) {
            Instruction move(InstructionOp::MOVE);
            move.args["target"] = defender_id;
            move.args["to"] = "GRAVEYARD";
            generated.push_back(move);
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    int BattleSystem::get_creature_power(const CardInstance& creature, const GameState& game_state,
                                         const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 0;
        int power = card_db.at(creature.card_id).power;

        // Apply active modifiers
        for (const auto& mod : game_state.active_modifiers) {
             (void)mod;
             // Logic for checking power modifier would go here (omitted for brevity, needs porting if present)
        }

        // Apply passive effects
        for (const auto& pe : game_state.passive_effects) {
            if (pe.type == PassiveType::POWER_MODIFIER) {
                // Simplified check: assume broad filter match or specific target
                // TODO: Full implementation of is_valid_target for passive effects
                if (pe.specific_targets.has_value()) {
                    for (int tid : pe.specific_targets.value()) {
                        if (tid == creature.instance_id) power += pe.value;
                    }
                }
            }
        }

        return power;
    }

}
