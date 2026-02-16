#include "command_system.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"
#include "engine/utils/target_utils.hpp"
#include "engine/systems/rules/condition_system.hpp"
#include "engine/infrastructure/data/card_registry.hpp"
#include "engine/utils/zone_utils.hpp"
#include <iostream>
#include <algorithm>
#include <set>

namespace dm::engine::systems {

    using namespace dm::core;

    Zone parse_zone_string(const std::string& zone_str);

    // Count cards logic (same as before)
    static int count_matching_cards(GameState& state, const CommandDef& cmd, PlayerID player_id, std::map<std::string, int>& execution_context) {
        std::vector<PlayerID> players_to_check;
        if (cmd.target_group == TargetScope::PLAYER_SELF || cmd.target_group == TargetScope::SELF) {
            players_to_check.push_back(player_id);
        } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
            players_to_check.push_back(1 - player_id);
        } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
            players_to_check.push_back(player_id);
            players_to_check.push_back(1 - player_id);
        }

        if (players_to_check.empty()) {
            players_to_check.push_back(player_id);
        }

        core::FilterDef filter = cmd.target_filter;
        filter.count.reset();
        if (filter.zones.empty()) {
            filter.zones.push_back("BATTLE_ZONE");
        }

        const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();
        int total = 0;

        for (PlayerID pid : players_to_check) {
            for (const auto& zone_str : filter.zones) {
                Zone zone_enum = parse_zone_string(zone_str);
                const std::vector<CardInstance>* container = nullptr;
                switch (zone_enum) {
                    case Zone::HAND: container = &state.players[pid].hand; break;
                    case Zone::MANA: container = &state.players[pid].mana_zone; break;
                    case Zone::BATTLE: container = &state.players[pid].battle_zone; break;
                    case Zone::GRAVEYARD: container = &state.players[pid].graveyard; break;
                    case Zone::SHIELD: container = &state.players[pid].shield_zone; break;
                    case Zone::DECK: container = &state.players[pid].deck; break;
                    default: break;
                }
                if (!container) continue;

                for (const auto& card : *container) {
                    auto def_it = card_db.find(card.card_id);
                    if (def_it != card_db.end()) {
                        if (dm::engine::utils::TargetUtils::is_valid_target(card, def_it->second, filter, state, player_id, pid, false, &execution_context)) {
                            total++;
                        }
                    } else if (card.card_id == 0) {
                        if (dm::engine::utils::TargetUtils::is_valid_target(card, CardDefinition(), filter, state, player_id, pid, false, &execution_context)) {
                            total++;
                        }
                    }
                }
            }
        }
        return total;
    }

    Zone parse_zone_string(const std::string& zone_str) {
        if (zone_str == "DECK") return Zone::DECK;
        if (zone_str == "HAND") return Zone::HAND;
        if (zone_str == "MANA" || zone_str == "MANA_ZONE") return Zone::MANA;
        if (zone_str == "BATTLE" || zone_str == "BATTLE_ZONE") return Zone::BATTLE;
        if (zone_str == "GRAVEYARD") return Zone::GRAVEYARD;
        if (zone_str == "SHIELD" || zone_str == "SHIELD_ZONE") return Zone::SHIELD;
        return Zone::GRAVEYARD;
    }

    int CommandSystem::resolve_amount(const CommandDef& cmd, const std::map<std::string, int>& execution_context) {
        if (!cmd.input_value_key.empty()) {
            auto it = execution_context.find(cmd.input_value_key);
            if (it != execution_context.end()) {
                return it->second;
            }
        }
        return cmd.amount;
    }

    void CommandSystem::execute_command(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        // Legacy support: Generate instructions and execute via temporary pipeline
        auto instructions = generate_instructions(state, cmd, source_instance_id, player_id, execution_context);

        dm::engine::systems::PipelineExecutor pipeline;
        const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();

        // Seed pipeline context with current execution_context
        for (const auto& kv : execution_context) {
            pipeline.set_context_var(kv.first, kv.second);
        }

        pipeline.execute(instructions, state, card_db);

        // Write back context output if needed (simple int/string/vec mapping)
        // PipelineExecutor context uses variants, simplified back mapping:
        for (const auto& kv : pipeline.context) {
            if (std::holds_alternative<int>(kv.second)) {
                execution_context[kv.first] = std::get<int>(kv.second);
            }
        }
    }

    std::vector<Instruction> CommandSystem::generate_instructions(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        std::vector<Instruction> out;

        // Ensure defaults are loaded
        dm::engine::rules::ConditionSystem::instance().initialize_defaults();

        switch (cmd.type) {
            // Primitive Commands
            case core::CommandType::TRANSITION:
            case core::CommandType::MUTATE:
            case core::CommandType::FLOW:
            case core::CommandType::QUERY:
            case core::CommandType::SHUFFLE_DECK:
                generate_primitive_instructions(out, state, cmd, source_instance_id, player_id, execution_context);
                break;

            case core::CommandType::MANA_CHARGE:
                {
                    nlohmann::json args;
                    args["card"] = (cmd.instance_id != -1) ? cmd.instance_id : source_instance_id;
                    Instruction inst(InstructionOp::GAME_ACTION, args);
                    inst.args["type"] = "MANA_CHARGE";
                    out.push_back(inst);
                }
                break;

            // Macro Commands
            case core::CommandType::DRAW_CARD:
            case core::CommandType::BOOST_MANA:
            case core::CommandType::ADD_MANA:
            case core::CommandType::DESTROY:
            case core::CommandType::DISCARD:
            case core::CommandType::TAP:
            case core::CommandType::UNTAP:
            case core::CommandType::RETURN_TO_HAND:
            case core::CommandType::BREAK_SHIELD:
            case core::CommandType::POWER_MOD:
            case core::CommandType::ADD_KEYWORD:
            case core::CommandType::SEARCH_DECK:
            case core::CommandType::SEND_TO_MANA:
                generate_macro_instructions(out, state, cmd, source_instance_id, player_id, execution_context);
                break;

            default:
                break;
        }
        return out;
    }

    void CommandSystem::generate_primitive_instructions(std::vector<Instruction>& out, GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        if (cmd.type == core::CommandType::FLOW) {
            bool cond_result = true;
            if (cmd.condition.has_value()) {
                 const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();
                 cond_result = dm::engine::rules::ConditionSystem::instance().evaluate_def(state, cmd.condition.value(), source_instance_id, card_db, execution_context);
            }

            const auto& branch = cond_result ? cmd.if_true : cmd.if_false;
            for (const auto& child_cmd : branch) {
                auto sub = generate_instructions(state, child_cmd, source_instance_id, player_id, execution_context);
                out.insert(out.end(), sub.begin(), sub.end());
            }

        } else if (cmd.type == core::CommandType::TRANSITION) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            Zone from_z = parse_zone_string(cmd.from_zone);
            std::string to_z_str = cmd.to_zone; // Pass string to instruction

            if (targets.empty() && cmd.amount > 0 && from_z != Zone::GRAVEYARD) {
                int count = resolve_amount(cmd, execution_context);
                if (from_z == Zone::DECK) {
                     Instruction move(InstructionOp::MOVE);
                     move.args["target"] = "DECK_TOP";
                     move.args["count"] = count;
                     move.args["to"] = to_z_str;
                     out.push_back(move);

                     if (!cmd.output_value_key.empty()) {
                         Instruction calc(InstructionOp::MATH);
                         calc.args["lhs"] = count;
                         calc.args["op"] = "+";
                         calc.args["rhs"] = 0;
                         calc.args["out"] = cmd.output_value_key;
                         out.push_back(calc);
                     }
                     return;
                }
            }

            for (int target_id : targets) {
                Instruction move(InstructionOp::MOVE);
                move.args["target"] = target_id;
                move.args["to"] = to_z_str;
                out.push_back(move);
            }

            if (!cmd.output_value_key.empty()) {
                 Instruction calc(InstructionOp::MATH);
                 calc.args["lhs"] = (int)targets.size();
                 calc.args["op"] = "+";
                 calc.args["rhs"] = 0;
                 calc.args["out"] = cmd.output_value_key;
                 out.push_back(calc);
            }

        } else if (cmd.type == core::CommandType::QUERY) {
             std::string query_type = cmd.str_param.empty() ? "SELECT_TARGET" : cmd.str_param;
             if (query_type == "CARDS_MATCHING_FILTER") {
                 int count = count_matching_cards(state, cmd, player_id, execution_context);
                 if (!cmd.output_value_key.empty()) {
                     Instruction calc(InstructionOp::MATH);
                     calc.args["lhs"] = count;
                     calc.args["op"] = "+";
                     calc.args["rhs"] = 0;
                     calc.args["out"] = cmd.output_value_key;
                     out.push_back(calc);
                 }
            } else if (query_type == "MANA_CIVILIZATION_COUNT") {
                 Instruction get(InstructionOp::GET_STAT);
                 get.args["stat"] = "MANA_CIVILIZATION_COUNT";
                 get.args["out"] = cmd.output_value_key.empty() ? "$temp" : cmd.output_value_key;
                 out.push_back(get);
            } else {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                if (targets.empty()) {
                    if (!cmd.output_value_key.empty()) {
                         // Set output to 0/empty?
                         // For query, usually we select from options.
                    }
                    return;
                }

                int amount = resolve_amount(cmd, execution_context);

                Instruction select(InstructionOp::SELECT);
                // We need to pass targets specifically.
                // But Pipeline SELECT usually takes filter.
                // We can synthesize a filter or use valid_targets override if we extended Pipeline.
                // Current Pipeline `handle_select` doesn't support `valid_targets` arg override well (logic in game_logic handled it).
                // Actually `PipelineExecutor::handle_select` logic:
                // It iterates filter.
                // BUT `GameLogicSystem::handle_select_target` supports `valid_targets`.
                // Let's use `GAME_ACTION` "SELECT_TARGET" which we implemented in PlaySystem/GameLogicSystem.

                nlohmann::json args;
                args["type"] = "SELECT_TARGET";
                args["valid_targets"] = targets;
                args["count"] = amount;
                args["out"] = cmd.output_value_key;

                Instruction inst(InstructionOp::GAME_ACTION, args);
                out.push_back(inst);
            }

        } else if (cmd.type == core::CommandType::MUTATE) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int val = resolve_amount(cmd, execution_context);

            Instruction modify(InstructionOp::MODIFY);
            modify.args["type"] = cmd.mutation_kind;
            modify.args["value"] = val;
            modify.args["str_value"] = cmd.str_param;

            for (int target_id : targets) {
                modify.args["target"] = target_id;
                out.push_back(modify);
            }

        } else if (cmd.type == core::CommandType::SHUFFLE_DECK) {
             Instruction modify(InstructionOp::MODIFY);
             modify.args["type"] = "SHUFFLE";
             modify.args["target"] = "DECK";
             out.push_back(modify);
        }
    }

    void CommandSystem::generate_macro_instructions(std::vector<Instruction>& out, GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        int count = resolve_amount(cmd, execution_context);

        // Example: DRAW_CARD
        if (cmd.type == core::CommandType::DRAW_CARD) {
            std::string out_key = cmd.output_value_key.empty() ? "$draw_choice" : cmd.output_value_key;
            std::string count_val_key = "";

            if (cmd.up_to && count > 0) {
                 Instruction select(InstructionOp::WAIT_INPUT);
                 select.args["query_type"] = "SELECT_NUMBER";
                 select.args["min"] = 0;
                 select.args["max"] = count;
                 select.args["out"] = out_key;
                 out.push_back(select);

                 count_val_key = out_key;
            }

            Instruction move(InstructionOp::MOVE);
            move.args["target"] = "DECK_TOP";
            if (!count_val_key.empty()) {
                move.args["count"] = count_val_key;
            } else {
                move.args["count"] = count;
            }
            move.args["to"] = "HAND";
            out.push_back(move);

            if (!cmd.output_value_key.empty() && count_val_key.empty()) {
                 Instruction calc(InstructionOp::MATH);
                 calc.args["lhs"] = count;
                 calc.args["op"] = "+";
                 calc.args["rhs"] = 0;
                 calc.args["out"] = cmd.output_value_key;
                 out.push_back(calc);
            }
            return;
        }

        // Example: DESTROY
        if (cmd.type == core::CommandType::DESTROY) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                Instruction move(InstructionOp::MOVE);
                move.args["target"] = target_id;
                move.args["to"] = "GRAVEYARD";
                out.push_back(move);
            }
             if (!cmd.output_value_key.empty()) {
                 Instruction calc(InstructionOp::MATH);
                 calc.args["lhs"] = (int)targets.size();
                 calc.args["op"] = "+";
                 calc.args["rhs"] = 0;
                 calc.args["out"] = cmd.output_value_key;
                 out.push_back(calc);
            }
            return;
        }

        // Fallback: Use Primitive Logic via dispatch if macros map cleanly to Primitives
        // For BOOST_MANA, DISCARD, etc.
        // Re-use primitive generator logic by creating a primitive-like structure?
        // Or just implement them.

        if (cmd.type == core::CommandType::BOOST_MANA || cmd.type == core::CommandType::ADD_MANA) {
             Instruction move(InstructionOp::MOVE);
             move.args["target"] = "DECK_TOP";
             move.args["count"] = count;
             move.args["to"] = "MANA";
             out.push_back(move);
        }
        else if (cmd.type == core::CommandType::DISCARD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                Instruction move(InstructionOp::MOVE);
                move.args["target"] = target_id;
                move.args["to"] = "GRAVEYARD";
                out.push_back(move);
            }
        }
        else if (cmd.type == core::CommandType::TAP || cmd.type == core::CommandType::UNTAP) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            std::string type = (cmd.type == core::CommandType::TAP) ? "TAP" : "UNTAP";
            for (int target_id : targets) {
                Instruction mod(InstructionOp::MODIFY);
                mod.args["type"] = type;
                mod.args["target"] = target_id;
                out.push_back(mod);
            }
        }
        else if (cmd.type == core::CommandType::BREAK_SHIELD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            // BREAK_SHIELD involves checking triggers.
            // Use GAME_ACTION "BREAK_SHIELD" which is handled by ShieldSystem
            nlohmann::json args;
            args["type"] = "BREAK_SHIELD";
            args["shields"] = targets;
            args["source_id"] = source_instance_id;

            Instruction inst(InstructionOp::GAME_ACTION, args);
            out.push_back(inst);
        }
    }

    std::vector<int> CommandSystem::resolve_targets(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        // ... (Same implementation as before)
        std::vector<int> targets;
        std::vector<PlayerID> players_to_check;

        if (cmd.target_group == TargetScope::PLAYER_SELF) {
            players_to_check.push_back(player_id);
        } else if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
            players_to_check.push_back(1 - player_id);
        } else if (cmd.target_group == TargetScope::ALL_PLAYERS) {
            players_to_check.push_back(player_id);
            players_to_check.push_back(1 - player_id);
        } else if (cmd.target_group == TargetScope::SELF) {
             players_to_check.push_back(player_id);
        }

        const auto& filter = cmd.target_filter;
        const auto& card_db = dm::engine::infrastructure::CardRegistry::get_all_definitions();

        for (PlayerID pid : players_to_check) {
            if (filter.zones.empty()) {
                if (cmd.target_group == TargetScope::SELF && source_instance_id != -1) {
                     CardInstance* inst = state.get_card_instance(source_instance_id);
                     if (inst && card_db.find(inst->card_id) != card_db.end()) {
                        if (dm::engine::utils::TargetUtils::is_valid_target(
                                *inst,
                                card_db.at(inst->card_id),
                                filter, state, player_id, pid, false, &execution_context)) {
                            targets.push_back(source_instance_id);
                        }
                     }
                }
                continue;
            }

            for (const std::string& zone_str : filter.zones) {
                Zone zone_enum = parse_zone_string(zone_str);

                const std::vector<CardInstance>* container = nullptr;
                switch (zone_enum) {
                    case Zone::HAND: container = &state.players[pid].hand; break;
                    case Zone::MANA: container = &state.players[pid].mana_zone; break;
                    case Zone::BATTLE: container = &state.players[pid].battle_zone; break;
                    case Zone::GRAVEYARD: container = &state.players[pid].graveyard; break;
                    case Zone::SHIELD: container = &state.players[pid].shield_zone; break;
                    case Zone::DECK: container = &state.players[pid].deck; break;
                    default: break;
                }

                if (container) {
                    for (const auto& card : *container) {
                         if (card_db.find(card.card_id) != card_db.end()) {
                             const auto& def = card_db.at(card.card_id);
                             if (dm::engine::utils::TargetUtils::is_valid_target(
                                     card, def, filter, state, player_id, pid, false, &execution_context)) {
                                 targets.push_back(card.instance_id);
                             }
                         }
                    }
                }
            }
        }

        if (filter.count.has_value()) {
            int n = filter.count.value();
            if (targets.size() > (size_t)n) {
                targets.resize(n);
            }
        }

        return targets;
    }

}
