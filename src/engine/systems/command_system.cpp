
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
        // Ensure defaults are loaded (singleton lazy init might need explicit call if not done elsewhere)
        ConditionSystem::instance().initialize_defaults();

        // Support both primitives and formerly "macro" commands as direct execution paths
        switch (cmd.type) {
            case core::CommandType::TRANSITION:
            case core::CommandType::MUTATE:
            case core::CommandType::FLOW:
            case core::CommandType::QUERY:
            case core::CommandType::SHUFFLE_DECK:
            case core::CommandType::DRAW_CARD:
            case core::CommandType::MANA_CHARGE:
            case core::CommandType::DESTROY:
            case core::CommandType::DISCARD:
            case core::CommandType::TAP:
            case core::CommandType::UNTAP:
            case core::CommandType::RETURN_TO_HAND:
            case core::CommandType::BREAK_SHIELD:
            case core::CommandType::POWER_MOD:
            case core::CommandType::ADD_KEYWORD:
            case core::CommandType::SEARCH_DECK:
                execute_primitive(state, cmd, source_instance_id, player_id, execution_context);
                break;
            case core::CommandType::TAP:
            case core::CommandType::UNTAP:
            case core::CommandType::DESTROY:
            case core::CommandType::DISCARD:
            case core::CommandType::RETURN_TO_HAND:
            case core::CommandType::MANA_CHARGE:
            case core::CommandType::DRAW_CARD:
            case core::CommandType::BREAK_SHIELD:
            case core::CommandType::POWER_MOD:
            case core::CommandType::ADD_KEYWORD:
            case core::CommandType::SEARCH_DECK:
                 expand_and_execute_macro(state, cmd, source_instance_id, player_id, execution_context);
                 break;
            default:
                break;
        }
    }

    void CommandSystem::execute_primitive(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        if (cmd.type == core::CommandType::FLOW) {
            // FLOW Primitive: Evaluate condition and execute branch
            bool cond_result = true;
            if (cmd.condition.has_value()) {
                 const auto& card_db = CardRegistry::get_all_definitions();
                 // ConditionSystem expects map<string, int>
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

            Zone from_z = parse_zone_string(cmd.from_zone);
            Zone to_z = parse_zone_string(cmd.to_zone);

            // Special Handling for DRAW_CARD style transitions (Top of Deck / Implicit Selection)
            if (targets.empty() && cmd.amount > 0 && from_z != Zone::GRAVEYARD) {
                // If specific zone is requested but no specific targets (e.g. DRAW 2 from DECK)
                int count = resolve_amount(cmd, execution_context);

                // Simplification: If from_zone is DECK and targets empty -> Take top N.
                if (from_z == Zone::DECK) {
                     int available = std::min((int)state.players[player_id].deck.size(), count);
                     for (int i = 0; i < available; ++i) {
                         if (!state.players[player_id].deck.empty()) {
                            int cid = state.players[player_id].deck.back().instance_id;
                            TransitionCommand trans(cid, from_z, to_z, player_id);
                            trans.execute(state);
                         }
                     }
                }
            }

            // Normal Target-based Transition
            int moved_count = 0;
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    Zone actual_from = from_z;
                    // Auto-detect source zone if unknown (GRAVEYARD is default/error return from parse)
                    if (actual_from == Zone::GRAVEYARD && cmd.from_zone != "GRAVEYARD") {
                        // Check common zones
                        if (ZoneUtils::card_in_zone(state.players[inst->owner].battle_zone, target_id)) actual_from = Zone::BATTLE;
                        else if (ZoneUtils::card_in_zone(state.players[inst->owner].mana_zone, target_id)) actual_from = Zone::MANA;
                        else if (ZoneUtils::card_in_zone(state.players[inst->owner].shield_zone, target_id)) actual_from = Zone::SHIELD;
                        else if (ZoneUtils::card_in_zone(state.players[inst->owner].hand, target_id)) actual_from = Zone::HAND;
                    }

                    TransitionCommand trans(target_id, actual_from, to_z, inst->owner);
                    trans.execute(state);
                    moved_count++;
                }
            }
            if (!cmd.output_value_key.empty()) {
                execution_context[cmd.output_value_key] = moved_count;
            }

        } else if (cmd.type == core::CommandType::QUERY) {
             // Map cmd parameters to Query params
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
            MutateCommand::MutationType m_type = MutateCommand::MutationType::TAP; // Default init

            if (cmd.mutation_kind == "TAP") m_type = MutateCommand::MutationType::TAP;
            else if (cmd.mutation_kind == "UNTAP") m_type = MutateCommand::MutationType::UNTAP;
            else if (cmd.mutation_kind == "POWER_MOD") m_type = MutateCommand::MutationType::POWER_MOD;
            else if (cmd.mutation_kind == "ADD_KEYWORD") m_type = MutateCommand::MutationType::ADD_KEYWORD;
            else if (cmd.mutation_kind == "REMOVE_KEYWORD") m_type = MutateCommand::MutationType::REMOVE_KEYWORD;
            else if (cmd.mutation_kind == "ADD_PASSIVE_EFFECT") m_type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
            else if (cmd.mutation_kind == "ADD_COST_MODIFIER") m_type = MutateCommand::MutationType::ADD_COST_MODIFIER;
            else if (cmd.mutation_kind == "ADD_PENDING_EFFECT") m_type = MutateCommand::MutationType::ADD_PENDING_EFFECT;
            else if (cmd.mutation_kind == "RESET_INSTANCE") m_type = MutateCommand::MutationType::RESET_INSTANCE;
            else if (cmd.mutation_kind == "POWER_SET") m_type = MutateCommand::MutationType::POWER_SET;
            else if (cmd.mutation_kind == "HEAL") m_type = MutateCommand::MutationType::HEAL;
            else valid_mutation = false;

            if (valid_mutation) {
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, m_type, val, cmd.str_param);
                    mutate.execute(state);
                }
            } else {
                std::cerr << "Warning: Unknown mutation kind: " << cmd.mutation_kind << std::endl;
            }

        // --- Former Macros / Primitives ---

        } else if (cmd.type == core::CommandType::TAP) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                MutateCommand mutate(target_id, MutateCommand::MutationType::TAP);
                mutate.execute(state);
            }
        } else if (cmd.type == core::CommandType::UNTAP) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                MutateCommand mutate(target_id, MutateCommand::MutationType::UNTAP);
                mutate.execute(state);
            }
        } else if (cmd.type == core::CommandType::RETURN_TO_HAND) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int returned = 0;
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                        Zone current_zone = Zone::BATTLE; // Default guess
                        PlayerID owner = inst->owner;
                        bool found = false;

                        // Optimized: Check zones if filter zones provided, otherwise standard search
                        if (ZoneUtils::card_in_zone(state.players[owner].battle_zone, target_id)) { current_zone = Zone::BATTLE; found = true; }
                        else if (ZoneUtils::card_in_zone(state.players[owner].mana_zone, target_id)) { current_zone = Zone::MANA; found = true; }
                        else if (ZoneUtils::card_in_zone(state.players[owner].shield_zone, target_id)) { current_zone = Zone::SHIELD; found = true; }
                        else if (ZoneUtils::card_in_zone(state.players[owner].graveyard, target_id)) { current_zone = Zone::GRAVEYARD; found = true; }

                        if (found) {
                            TransitionCommand trans(target_id, current_zone, Zone::HAND, owner);
                            trans.execute(state);
                            returned++;
                        }
                }
            }
            if (!cmd.output_value_key.empty()) execution_context[cmd.output_value_key] = returned;

        } else if (cmd.type == core::CommandType::DRAW_CARD) {
            int count = resolve_amount(cmd, execution_context);
            int drawn = 0;
            for (int i = 0; i < count; ++i) {
                if (!state.players[player_id].deck.empty()) {
                    int card_inst_id = state.players[player_id].deck.back().instance_id;
                    TransitionCommand trans(card_inst_id, Zone::DECK, Zone::HAND, player_id);
                    trans.execute(state);
                    drawn++;
                }
            }
            if (!cmd.output_value_key.empty()) execution_context[cmd.output_value_key] = drawn;

        } else if (cmd.type == core::CommandType::MANA_CHARGE) {
            int count = resolve_amount(cmd, execution_context);
            int charged = 0;
            for (int i = 0; i < count; ++i) {
                if (!state.players[player_id].deck.empty()) {
                    int card_inst_id = state.players[player_id].deck.back().instance_id;
                    TransitionCommand trans(card_inst_id, Zone::DECK, Zone::MANA, player_id);
                    trans.execute(state);
                    charged++;
                }
            }
            if (!cmd.output_value_key.empty()) execution_context[cmd.output_value_key] = charged;

        } else if (cmd.type == core::CommandType::DESTROY) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int destroyed = 0;
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    TransitionCommand trans(target_id, Zone::BATTLE, Zone::GRAVEYARD, inst->owner);
                    trans.execute(state);
                    destroyed++;
                }
            }
            if (!cmd.output_value_key.empty()) execution_context[cmd.output_value_key] = destroyed;

        } else if (cmd.type == core::CommandType::DISCARD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int discarded = 0;
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    TransitionCommand trans(target_id, Zone::HAND, Zone::GRAVEYARD, inst->owner);
                    trans.execute(state);
                    discarded++;
                }
            }
            if (!cmd.output_value_key.empty()) execution_context[cmd.output_value_key] = discarded;

        } else if (cmd.type == core::CommandType::BREAK_SHIELD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    TransitionCommand trans(target_id, Zone::SHIELD, Zone::HAND, inst->owner);
                    trans.execute(state);
                }
            }
        } else if (cmd.type == core::CommandType::POWER_MOD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            int val = resolve_amount(cmd, execution_context);
            for (int target_id : targets) {
                MutateCommand mutate(target_id, MutateCommand::MutationType::POWER_MOD, val);
                mutate.execute(state);
            }
        } else if (cmd.type == core::CommandType::ADD_KEYWORD) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
            for (int target_id : targets) {
                MutateCommand mutate(target_id, MutateCommand::MutationType::ADD_KEYWORD, 0, cmd.str_param);
                mutate.execute(state);
            }
        } else if (cmd.type == core::CommandType::SEARCH_DECK) {
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
        }
    }

    void CommandSystem::expand_and_execute_macro(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        int count = resolve_amount(cmd, execution_context);

        switch (cmd.type) {
            case core::CommandType::DRAW_CARD: {
                int drawn = 0;
                for (int i = 0; i < count; ++i) {
                     const auto& deck = state.players[player_id].deck;
                     if (!deck.empty()) {
                         int card_inst_id = deck.back().instance_id;
                         TransitionCommand trans(card_inst_id, Zone::DECK, Zone::HAND, player_id);
                         trans.execute(state);
                         drawn++;
                     }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = drawn;
                }
                break;
            }
            case core::CommandType::MANA_CHARGE: {
                 int charged = 0;
                 for (int i = 0; i < count; ++i) {
                     const auto& deck = state.players[player_id].deck;
                     if (!deck.empty()) {
                         int card_inst_id = deck.back().instance_id;
                         TransitionCommand trans(card_inst_id, Zone::DECK, Zone::MANA, player_id);
                         trans.execute(state);
                         charged++;
                     }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = charged;
                }
                break;
            }
            case core::CommandType::DESTROY: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int destroyed = 0;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         TransitionCommand trans(target_id, Zone::BATTLE, Zone::GRAVEYARD, inst->owner);
                         trans.execute(state);
                         destroyed++;
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = destroyed;
                }
                break;
            }
            case core::CommandType::DISCARD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int discarded = 0;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                        TransitionCommand trans(target_id, Zone::HAND, Zone::GRAVEYARD, inst->owner);
                        trans.execute(state);
                        discarded++;
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = discarded;
                }
                break;
            }
            case core::CommandType::TAP: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, MutateCommand::MutationType::TAP);
                    mutate.execute(state);
                }
                break;
            }
            case core::CommandType::UNTAP: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, MutateCommand::MutationType::UNTAP);
                    mutate.execute(state);
                }
                break;
            }
            case core::CommandType::RETURN_TO_HAND: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int returned = 0;
                Zone from_z = cmd.from_zone.empty() ? Zone::GRAVEYARD : parse_zone_string(cmd.from_zone); // GRAVEYARD as sentinel for unknown

                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         Zone current_zone = Zone::BATTLE; // Default guess
                         bool found = false;

                         if (from_z != Zone::GRAVEYARD) {
                             current_zone = from_z;
                             found = true;
                         } else {
                             PlayerID owner = inst->owner;
                             // Fast check
                             if (ZoneUtils::card_in_zone(state.players[owner].battle_zone, target_id)) { current_zone = Zone::BATTLE; found = true; }
                             else if (ZoneUtils::card_in_zone(state.players[owner].mana_zone, target_id)) { current_zone = Zone::MANA; found = true; }
                             else if (ZoneUtils::card_in_zone(state.players[owner].shield_zone, target_id)) { current_zone = Zone::SHIELD; found = true; }
                             else if (ZoneUtils::card_in_zone(state.players[owner].graveyard, target_id)) { current_zone = Zone::GRAVEYARD; found = true; }
                         }

                         if (found) {
                             TransitionCommand trans(target_id, current_zone, Zone::HAND, inst->owner);
                             trans.execute(state);
                             returned++;
                         }
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = returned;
                }
                break;
            }
            case core::CommandType::BREAK_SHIELD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int broken = 0;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         // Standard Break: Shield -> Hand
                         TransitionCommand trans(target_id, Zone::SHIELD, Zone::HAND, inst->owner);
                         trans.execute(state);
                         broken++;

                         // Note: This command implementation does NOT handle S-Triggers.
                         // That requires EffectResolver flow. CommandSystem is low-level state mutation.
                    }
                }
                break;
            }
            case core::CommandType::POWER_MOD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int val = resolve_amount(cmd, execution_context);
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, MutateCommand::MutationType::POWER_MOD, val);
                    mutate.execute(state);
                }
                break;
            }
            case core::CommandType::ADD_KEYWORD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, MutateCommand::MutationType::ADD_KEYWORD, 0, cmd.str_param);
                    mutate.execute(state);
                }
                break;
            }
            case core::CommandType::SEARCH_DECK: {
                 // 1. Resolve Targets (implies Query/Select done implicitly or via filter)
                 // If TargetScope is TARGET_SELECT, this implies we need user input, but CommandSystem executes immediately.
                 // If it's a test/AI scenario, we assume targets are chosen.
                 // For now, we take `resolve_targets` as the "Selected Cards".
                 // If no targets and Filter exists, we pick from Deck matching Filter (Deterministic/First valid).

                 std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);

                 // 2. Move to Hand (or Destination)
                 Zone dest_zone = cmd.to_zone.empty() ? Zone::HAND : parse_zone_string(cmd.to_zone);

                 for (int target_id : targets) {
                     CardInstance* inst = state.get_card_instance(target_id);
                     if (inst) {
                         TransitionCommand trans(target_id, Zone::DECK, dest_zone, inst->owner);
                         trans.execute(state);
                     }
                 }

                 // 3. Shuffle
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
                // For now, naive truncation (first N).
                // Ideally this should be random or selected, but CommandSystem executes outcome.
                targets.resize(n);
            }
        }

        return targets;
    }

}
