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
            case core::CommandType::SELECT_NUMBER:
            case core::CommandType::SELECT_TARGET:
                generate_primitive_instructions(out, state, cmd, source_instance_id, player_id, execution_context);
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

            int count = resolve_amount(cmd, execution_context);
            if (targets.empty() && (count > 0 || !cmd.input_value_key.empty()) && from_z != Zone::GRAVEYARD) {
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
                } else if (from_z == Zone::HAND) {
                     // Implicit Selection for Hand -> Zone
                     Instruction select(InstructionOp::SELECT);
                     core::FilterDef filter;
                     filter.zones.push_back("HAND");
                     filter.owner = "SELF"; // Assume self unless specified
                     // TODO: Propagate other filters from cmd.target_filter?

                     select.args["filter"] = filter;

                     // Handle dynamic count if input_value_key was used
                     if (!cmd.input_value_key.empty()) {
                         select.args["count"] = "$" + cmd.input_value_key;
                     } else {
                         select.args["count"] = count;
                     }

                     std::string implicit_var = "$implicit_sel_" + std::to_string(cmd.instance_id);
                     select.args["out"] = implicit_var;
                     out.push_back(select);

                     Instruction move(InstructionOp::MOVE);
                     move.args["target"] = implicit_var;
                     move.args["to"] = to_z_str;
                     if (to_z_str == "DECK" || to_z_str == "DECK_BOTTOM") {
                         move.args["to_bottom"] = true;
                         move.args["to"] = "DECK";
                     }
                     out.push_back(move);

                     if (!cmd.output_value_key.empty()) {
                         // Pass implicit var as output? Or size?
                         // Usually commands output size or list.
                         // Let's copy implicit var to output key
                         // Actually, set_context_var/calc logic?
                         // Just alias it.
                         // But for now, returning is enough.
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
        } else if (cmd.type == core::CommandType::SELECT_NUMBER) {
             Instruction wait(InstructionOp::WAIT_INPUT);
             wait.args["query_type"] = "SELECT_NUMBER";
             if (cmd.target_filter.min_cost.has_value()) wait.args["min"] = cmd.target_filter.min_cost.value();

             if (cmd.target_filter.cost_ref.has_value()) {
                 std::string ref = cmd.target_filter.cost_ref.value();
                 if (!ref.empty() && ref[0] != '$') ref = "$" + ref;
                 wait.args["max"] = ref;
             } else if (cmd.target_filter.max_cost.has_value()) {
                 wait.args["max"] = cmd.target_filter.max_cost.value();
             } else if (cmd.amount > 0) {
                 wait.args["max"] = cmd.amount;
             }
             wait.args["out"] = cmd.output_value_key.empty() ? "$result" : cmd.output_value_key;
             out.push_back(wait);
        } else if (cmd.type == core::CommandType::SELECT_TARGET) {
             Instruction select(InstructionOp::SELECT);
             select.args["filter"] = cmd.target_filter;

             if (!cmd.input_value_key.empty()) {
                 select.args["count"] = "$" + cmd.input_value_key;
             } else {
                 select.args["count"] = cmd.amount;
             }
             select.args["out"] = cmd.output_value_key.empty() ? "$selection" : cmd.output_value_key;
             out.push_back(select);
        }
    }

    void CommandSystem::generate_macro_instructions(std::vector<Instruction>& out, GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        int count = resolve_amount(cmd, execution_context);

        // Example: DRAW_CARD
        if (cmd.type == core::CommandType::DRAW_CARD) {
            std::string draw_count_var = "";
            if (cmd.up_to) {
                 Instruction wait(InstructionOp::WAIT_INPUT);
                 wait.args["query_type"] = "SELECT_NUMBER";
                 wait.args["min"] = 0;
                 wait.args["max"] = count;

                 std::string var = cmd.output_value_key.empty() ? "$draw_amount" : cmd.output_value_key;
                 wait.args["out"] = var;
                 out.push_back(wait);

                 draw_count_var = var;
            }

            Instruction move(InstructionOp::MOVE);
            move.args["target"] = "DECK_TOP";
            if (!draw_count_var.empty()) {
                // Ensure variable reference has $
                if (draw_count_var[0] != '$') move.args["count"] = "$" + draw_count_var;
                else move.args["count"] = draw_count_var;
            } else if (!cmd.input_value_key.empty()) {
                // Dynamic amount from previous instruction
                move.args["count"] = "$" + cmd.input_value_key;
            } else {
                move.args["count"] = count;
            }
            move.args["to"] = "HAND";
            out.push_back(move);

            if (!cmd.output_value_key.empty() && !cmd.up_to) {
                 Instruction calc(InstructionOp::MATH);
                 calc.args["lhs"] = count; // Approximation
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
