
#include "command_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/utils/zone_utils.hpp"
#include <iostream>
#include <algorithm>
#include <random>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    Zone parse_zone_string(const std::string& zone_str) {
        if (zone_str == "DECK") return Zone::DECK;
        if (zone_str == "HAND") return Zone::HAND;
        if (zone_str == "MANA" || zone_str == "MANA_ZONE") return Zone::MANA;
        if (zone_str == "BATTLE" || zone_str == "BATTLE_ZONE") return Zone::BATTLE;
        if (zone_str == "GRAVEYARD") return Zone::GRAVEYARD;
        if (zone_str == "SHIELD" || zone_str == "SHIELD_ZONE") return Zone::SHIELD;
        if (zone_str == "BUFFER") return Zone::BUFFER;
        if (zone_str == "STACK") return Zone::STACK;
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
        ConditionSystem::instance().initialize_defaults();

        switch (cmd.type) {
            case core::CommandType::TRANSITION:
            case core::CommandType::MUTATE:
            case core::CommandType::FLOW:
            case core::CommandType::QUERY:
                execute_primitive(state, cmd, source_instance_id, player_id, execution_context);
                break;
            default:
                expand_and_execute_macro(state, cmd, source_instance_id, player_id, execution_context);
                break;
        }
    }

    // Helper to find which zone a card is in
    Zone find_card_zone(const GameState& state, int instance_id, PlayerID owner_id) {
         const auto& p = state.players[owner_id];
         auto check = [&](const std::vector<CardInstance>& vec, Zone z) {
             for(const auto& c : vec) if(c.instance_id == instance_id) return true;
             return false;
         };
         if (check(p.battle_zone, Zone::BATTLE)) return Zone::BATTLE;
         if (check(p.mana_zone, Zone::MANA)) return Zone::MANA;
         if (check(p.shield_zone, Zone::SHIELD)) return Zone::SHIELD;
         if (check(p.hand, Zone::HAND)) return Zone::HAND;
         if (check(p.graveyard, Zone::GRAVEYARD)) return Zone::GRAVEYARD;
         if (check(p.effect_buffer, Zone::BUFFER)) return Zone::BUFFER;
         if (check(p.stack, Zone::STACK)) return Zone::STACK;
         // Deck usually not checked for generic instance ID lookup unless tracked
         return Zone::GRAVEYARD; // Fallback / Unknown
    }

    void CommandSystem::execute_primitive(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        if (cmd.type == core::CommandType::FLOW) {
            bool cond_result = true;
            if (cmd.condition.has_value()) {
                 const auto& card_db = CardRegistry::get_all_definitions();
                 cond_result = ConditionSystem::instance().evaluate_def(
                     state, cmd.condition.value(), source_instance_id, card_db, execution_context
                 );
            }

            const auto& branch = cond_result ? cmd.if_true : cmd.if_false;
            for (const auto& child_cmd : branch) {
                execute_command(state, child_cmd, source_instance_id, player_id, execution_context);
            }

        } else if (cmd.type == core::CommandType::TRANSITION) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int count = resolve_amount(cmd, execution_context);
            Zone explicit_from = cmd.from_zone.empty() ? Zone::GRAVEYARD : parse_zone_string(cmd.from_zone);
            // Note: If empty, we use GRAVEYARD as placeholder but we'll try to infer.
            bool infer_from = cmd.from_zone.empty();
            Zone to_z = parse_zone_string(cmd.to_zone);

            // Special Case: "Draw N" or "Mill N" (From DECK) where targets are not explicit
            if (targets.empty() && count > 0 && explicit_from == Zone::DECK) {
                // Auto-select top N cards from deck
                const auto& deck = state.players[player_id].deck;
                int available = deck.size();
                int to_move = std::min(count, available);
                // Deck is a vector where back() is Top.
                // We iterate from back.
                for (int i = 0; i < to_move; ++i) {
                     // Get current top (which changes as we pop, but TransitionCommand handles it by ID)
                     // Actually TransitionCommand finds by ID. If we use indices it might be safer?
                     // But here we just grab the ID of the top card.
                     if (state.players[player_id].deck.empty()) break;
                     int top_id = state.players[player_id].deck.back().instance_id;
                     TransitionCommand trans(top_id, Zone::DECK, to_z, player_id);
                     trans.execute(state);
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = to_move;
                }
                return;
            }

            // Standard Case: Transition specific targets
            int moved_count = 0;
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    Zone from_z = explicit_from;
                    if (infer_from) {
                        from_z = find_card_zone(state, target_id, inst->owner);
                    }

                    TransitionCommand trans(target_id, from_z, to_z, inst->owner);
                    trans.execute(state);
                    moved_count++;
                }
            }
            if (!cmd.output_value_key.empty()) {
                execution_context[cmd.output_value_key] = moved_count;
            }

        } else if (cmd.type == core::CommandType::QUERY) {
             std::string query_type = cmd.str_param.empty() ? "SELECT_TARGET" : cmd.str_param;
             std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);

             std::map<std::string, int> params;
             params["amount"] = resolve_amount(cmd, execution_context);

             QueryCommand query(query_type, targets, params);
             query.execute(state);

        } else if (cmd.type == core::CommandType::MUTATE) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int val = resolve_amount(cmd, execution_context);

            bool valid_mutation = true;
            MutateCommand::MutationType m_type = MutateCommand::MutationType::TAP;

            if (cmd.mutation_kind == "TAP") m_type = MutateCommand::MutationType::TAP;
            else if (cmd.mutation_kind == "UNTAP") m_type = MutateCommand::MutationType::UNTAP;
            else if (cmd.mutation_kind == "POWER_MOD") m_type = MutateCommand::MutationType::POWER_MOD;
            else if (cmd.mutation_kind == "ADD_KEYWORD") m_type = MutateCommand::MutationType::ADD_KEYWORD;
            else if (cmd.mutation_kind == "REMOVE_KEYWORD") m_type = MutateCommand::MutationType::REMOVE_KEYWORD;
            else if (cmd.mutation_kind == "ADD_PASSIVE_EFFECT") m_type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
            else if (cmd.mutation_kind == "ADD_COST_MODIFIER") m_type = MutateCommand::MutationType::ADD_COST_MODIFIER;
            else if (cmd.mutation_kind == "ADD_PENDING_EFFECT") m_type = MutateCommand::MutationType::ADD_PENDING_EFFECT;
            else if (cmd.mutation_kind == "REVOLUTION_CHANGE") m_type = MutateCommand::MutationType::ADD_PENDING_EFFECT; // Assume mapped? Or just ignore/warn?
            else valid_mutation = false;

            if (valid_mutation) {
                // If mutation is REVOLUTION_CHANGE, it likely maps to something specific or is handled differently?
                // For now, if string param is used for Mutation, we pass it.
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, m_type, val, cmd.str_param);
                    mutate.execute(state);
                }
            }
        }
    }

    void CommandSystem::expand_and_execute_macro(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        // Fallback for legacy types not yet normalized to TRANSITION/MUTATE
        // We can reuse TRANSITION logic for move-types here if we want to be clean,
        // or keep legacy macros for safety.
        // Given we updated action_mapper, these shouldn't be hit for NEW actions, but good for backward compat.

        int count = resolve_amount(cmd, execution_context);

        switch (cmd.type) {
            case core::CommandType::DRAW_CARD: {
                // Map to TRANSITION logic
                CommandDef t_cmd;
                t_cmd.type = core::CommandType::TRANSITION;
                t_cmd.from_zone = "DECK";
                t_cmd.to_zone = "HAND";
                t_cmd.amount = count;
                t_cmd.output_value_key = cmd.output_value_key;
                execute_primitive(state, t_cmd, source_instance_id, player_id, execution_context);
                break;
            }
            case core::CommandType::MANA_CHARGE: {
                 CommandDef t_cmd;
                 t_cmd.type = core::CommandType::TRANSITION;
                 t_cmd.from_zone = "DECK"; // Default legacy behavior
                 t_cmd.to_zone = "MANA";
                 t_cmd.amount = count;
                 t_cmd.output_value_key = cmd.output_value_key;
                 execute_primitive(state, t_cmd, source_instance_id, player_id, execution_context);
                 break;
            }
            case core::CommandType::DESTROY:
            case core::CommandType::DISCARD:
            case core::CommandType::RETURN_TO_HAND:
            case core::CommandType::BREAK_SHIELD: {
                // For these, we can construct a TRANSITION command, but we need to resolve targets first
                // to respect the filters in the original command.
                // Or we just call execute_primitive with type forced to TRANSITION and appropriate zones.

                CommandDef t_cmd = cmd; // Copy
                t_cmd.type = core::CommandType::TRANSITION;

                if (cmd.type == core::CommandType::DESTROY) t_cmd.to_zone = "GRAVEYARD";
                else if (cmd.type == core::CommandType::DISCARD) { t_cmd.to_zone = "GRAVEYARD"; t_cmd.from_zone = "HAND"; }
                else if (cmd.type == core::CommandType::RETURN_TO_HAND) t_cmd.to_zone = "HAND";
                else if (cmd.type == core::CommandType::BREAK_SHIELD) { t_cmd.to_zone = "HAND"; t_cmd.from_zone = "SHIELD"; }

                execute_primitive(state, t_cmd, source_instance_id, player_id, execution_context);
                break;
            }
            case core::CommandType::TAP: {
                CommandDef m_cmd = cmd;
                m_cmd.type = core::CommandType::MUTATE;
                m_cmd.mutation_kind = "TAP";
                execute_primitive(state, m_cmd, source_instance_id, player_id, execution_context);
                break;
            }
            case core::CommandType::UNTAP: {
                CommandDef m_cmd = cmd;
                m_cmd.type = core::CommandType::MUTATE;
                m_cmd.mutation_kind = "UNTAP";
                execute_primitive(state, m_cmd, source_instance_id, player_id, execution_context);
                break;
            }
            case core::CommandType::SEARCH_DECK: {
                 // Complex macro: Search -> Move -> Shuffle
                 std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                 Zone dest_zone = cmd.to_zone.empty() ? Zone::HAND : parse_zone_string(cmd.to_zone);

                 for (int target_id : targets) {
                     CardInstance* inst = state.get_card_instance(target_id);
                     if (inst) {
                         TransitionCommand trans(target_id, Zone::DECK, dest_zone, inst->owner);
                         trans.execute(state);
                     }
                 }
                 ShuffleCommand shuffle(player_id);
                 shuffle.execute(state);
                 break;
            }
            default:
                break;
        }
    }

    std::vector<int> CommandSystem::resolve_targets(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
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
        const auto& card_db = CardRegistry::get_all_definitions();

        for (PlayerID pid : players_to_check) {
            if (filter.zones.empty()) {
                if (cmd.target_group == TargetScope::SELF && source_instance_id != -1) {
                     CardInstance* inst = state.get_card_instance(source_instance_id);
                     if (inst && card_db.find(inst->card_id) != card_db.end()) {
                        if (dm::engine::TargetUtils::is_valid_target(
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
                    case Zone::BUFFER: container = &state.players[pid].effect_buffer; break;
                    default: break;
                }

                if (container) {
                    for (const auto& card : *container) {
                         if (card_db.find(card.card_id) != card_db.end()) {
                             const auto& def = card_db.at(card.card_id);
                             if (dm::engine::TargetUtils::is_valid_target(
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
