
#include "command_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/utils/zone_utils.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <fstream>
#include <filesystem>
#include <set>
#include "engine/systems/decision_maker.hpp"

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    Zone parse_zone_string(const std::string& zone_str);

    // Count cards that satisfy the command's target_filter across the selected players/zones.
    // Ignores filter.count so queries can report the true total.
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

        const auto& card_db = CardRegistry::get_all_definitions();
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
                        if (dm::engine::TargetUtils::is_valid_target(card, def_it->second, filter, state, player_id, pid, false, &execution_context)) {
                            total++;
                        }
                    } else if (card.card_id == 0) {
                        core::CardDefinition placeholder;
                        if (dm::engine::TargetUtils::is_valid_target(card, placeholder, filter, state, player_id, pid, false, &execution_context)) {
                            total++;
                        }
                    }
                }
            }
        }

        return total;
    }

    Zone parse_zone_string(const std::string& zone_str) {
        if (zone_str == "DECK" || zone_str == "DECK_BOTTOM") return Zone::DECK;
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

    bool CommandSystem::execute_command(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        // Ensure defaults are loaded (singleton lazy init might need explicit call if not done elsewhere)
        ConditionSystem::instance().initialize_defaults();

        switch (cmd.type) {
            // Primitive Commands
            case core::CommandType::TRANSITION:
            case core::CommandType::MUTATE:
            case core::CommandType::FLOW:
            case core::CommandType::QUERY:
            case core::CommandType::SHUFFLE_DECK:
                return execute_primitive(state, cmd, source_instance_id, player_id, execution_context);

            // Macro Commands
            case core::CommandType::DRAW_CARD:
            case core::CommandType::BOOST_MANA:
            case core::CommandType::ADD_MANA: // Alias to BOOST_MANA for simple count
            case core::CommandType::DESTROY:
            case core::CommandType::DISCARD:
            case core::CommandType::TAP:
            case core::CommandType::UNTAP:
            case core::CommandType::RETURN_TO_HAND:
            case core::CommandType::BREAK_SHIELD:
            case core::CommandType::POWER_MOD:
            case core::CommandType::ADD_KEYWORD:
            case core::CommandType::SEARCH_DECK:
            case core::CommandType::SEND_TO_MANA: // Handled as macro/transition
                return expand_and_execute_macro(state, cmd, source_instance_id, player_id, execution_context);

            default:
                break;
        }
        return true;
    }

    bool CommandSystem::execute_primitive(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
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
                if (!execute_command(state, child_cmd, source_instance_id, player_id, execution_context)) {
                    return false; // Suspended in flow branch
                }
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
            std::vector<int> moved_ids;
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

                    int dest_idx = -1;
                    if (cmd.to_zone == "DECK_BOTTOM") {
                         to_z = Zone::DECK;
                         dest_idx = 0;
                    }

                    TransitionCommand trans(target_id, actual_from, to_z, inst->owner, dest_idx);
                    trans.execute(state);
                    moved_count++;
                    moved_ids.push_back(target_id);
                }
            }
            if (!cmd.output_value_key.empty()) {
                execution_context[cmd.output_value_key] = moved_count;
                // Store card IDs in sequential keys
                if (!moved_ids.empty()) {
                    std::string ids_key = cmd.output_value_key + "_ids";
                    for (size_t i = 0; i < moved_ids.size(); ++i) {
                        execution_context[ids_key + "_" + std::to_string(i)] = moved_ids[i];
                    }
                    execution_context[ids_key + "_count"] = static_cast<int>(moved_ids.size());
                }
            }

        } else if (cmd.type == core::CommandType::QUERY) {
             // Map cmd parameters to Query params
             std::string query_type = cmd.str_param.empty() ? "SELECT_TARGET" : cmd.str_param;
             if (query_type == "CARDS_MATCHING_FILTER") {
                 int count = count_matching_cards(state, cmd, player_id, execution_context);
                 if (!cmd.output_value_key.empty()) {
                     execution_context[cmd.output_value_key] = count;
                 }
            } else if (query_type == "MANA_CIVILIZATION_COUNT") {
                 // Count distinct civilizations in player's mana zone
                 const auto& card_db = CardRegistry::get_all_definitions();
                 std::set<core::Civilization> civs_in_mana;
                 
                 for (const auto& card : state.players[player_id].mana_zone) {
                     auto def_it = card_db.find(card.card_id);
                     if (def_it != card_db.end()) {
                         for (const auto& civ : def_it->second.civilizations) {
                             civs_in_mana.insert(civ);
                         }
                     }
                 }
                 
                 int civ_count = static_cast<int>(civs_in_mana.size());
                 if (!cmd.output_value_key.empty()) {
                     execution_context[cmd.output_value_key] = civ_count;
                 }
            } else {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);

                // If there are no valid targets, do not create a pending query that waits for user input.
                // Instead, set output value (if requested) and continue execution immediately.
                if (targets.empty()) {
                    if (!cmd.output_value_key.empty()) {
                        execution_context[cmd.output_value_key] = 0;
                    }
                    return true;
                }

                std::map<std::string, int> params;
                params["amount"] = resolve_amount(cmd, execution_context);

                QueryCommand query(query_type, targets, params);
                query.execute(state);
                // Queries that require user input (like SELECT_TARGET) are typically handled via PendingEffect or
                // immediately resolved if AI/Random.
                // However, currently QueryCommand might set waiting_for_user_input.
                // If it does, we should interpret that as suspension?
                // Actually, existing Query logic in GameState might set 'waiting_for_user_input'
                // But let's check if we have a robust way to detect async. 
                // For now, assuming basic queries might be sync for AI or handled elsewhere.
                // But the critical fix is for DRAW_CARD optional which explicitly creates PendingEffect.
            }

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
                // std::cerr << "Warning: Unknown mutation kind: " << cmd.mutation_kind << std::endl;
            }
        } else if (cmd.type == core::CommandType::SHUFFLE_DECK) {
             ShuffleCommand shuffle(player_id);
             shuffle.execute(state);
        }
        return true;
    }

    bool CommandSystem::expand_and_execute_macro(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id, std::map<std::string, int>& execution_context) {
        int count = resolve_amount(cmd, execution_context);

        switch (cmd.type) {
            case core::CommandType::DRAW_CARD: {
                // Handle optional + up_to: Create pending effect for player choice
                if (cmd.optional && count > 0) {
                    // Create a pending effect for SELECT_NUMBER
                    core::PendingEffect pending(core::EffectType::SELECT_NUMBER, source_instance_id, player_id);
                    
                    // Set num_targets_needed for PendingEffectStrategy to generate actions
                    pending.num_targets_needed = count;  // max value player can select
                    
                    // Store min/max in execution_context
                    pending.execution_context = execution_context;
                    pending.execution_context["_min_select"] = 0; // optional means can choose 0
                    pending.execution_context["_max_select"] = count;
                    
                    // Create continuation effect with command to draw selected number
                    core::EffectDef continuation;
                    continuation.trigger = core::TriggerType::NONE;
                    
                    core::CommandDef draw_cmd;
                    draw_cmd.type = core::CommandType::DRAW_CARD;
                    draw_cmd.target_group = core::TargetScope::PLAYER_SELF;
                    draw_cmd.amount = 0; // Will be set by input_value_key
                    draw_cmd.input_value_key = "_selected_number";
                    if (!cmd.output_value_key.empty()) {
                        draw_cmd.output_value_key = cmd.output_value_key;
                    }
                    continuation.commands.push_back(draw_cmd);
                    
                    pending.effect_def = continuation;
                    
                    // Store output key for later retrieval
                    if (!cmd.output_value_key.empty()) {
                        pending.effect_def->condition.str_val = cmd.output_value_key;
                    }
                    
                    state.pending_effects.push_back(pending);
                    // std::cerr << "[CommandSystem] DRAW_CARD optional: Created SELECT_NUMBER pending effect (0-" << count << ")" << std::endl;
                    
                    // DEBUG: Log to file
                    try {
                        std::filesystem::create_directories("logs");
                        std::ofstream ofs("logs/select_number_debug.txt", std::ios::app);
                        if (ofs) {
                            ofs << "[CREATE_PENDING] type=SELECT_NUMBER controller=" << static_cast<int>(player_id)
                                << " num_targets_needed=" << count
                                << " src_iid=" << source_instance_id << "\n";
                        }
                    } catch(...) {}
                    
                    return false; // Don't draw now, wait for player input
                }
                
                // Normal draw (non-optional or count determined by input)
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
            case core::CommandType::BOOST_MANA:
            case core::CommandType::ADD_MANA: {
                 int charged = 0;
                 std::vector<int> charged_ids;
                 for (int i = 0; i < count; ++i) {
                     const auto& deck = state.players[player_id].deck;
                     if (!deck.empty()) {
                         int card_inst_id = deck.back().instance_id;
                         TransitionCommand trans(card_inst_id, Zone::DECK, Zone::MANA, player_id);
                         trans.execute(state);
                         charged++;
                         charged_ids.push_back(card_inst_id);
                     }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = charged;
                    // Store card IDs in sequential keys
                    if (!charged_ids.empty()) {
                        std::string ids_key = cmd.output_value_key + "_ids";
                        for (size_t i = 0; i < charged_ids.size(); ++i) {
                            execution_context[ids_key + "_" + std::to_string(i)] = charged_ids[i];
                        }
                        execution_context[ids_key + "_count"] = static_cast<int>(charged_ids.size());
                    }
                }
                break;
            }
            case core::CommandType::SEND_TO_MANA: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int moved = 0;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         // Move to Mana Zone
                         TransitionCommand trans(target_id, Zone::BATTLE, Zone::MANA, inst->owner);
                         // Note: TransitionCommand will update from_zone internally if it's different
                         // But we should try to infer. Currently TransitionCommand expects explicit from.
                         // But for generic SEND_TO_MANA, from_zone might be implicit.

                         // Better: Use CommandType::TRANSITION logic helper or just infer here:
                         Zone from_z = Zone::BATTLE;
                         if (ZoneUtils::card_in_zone(state.players[inst->owner].battle_zone, target_id)) from_z = Zone::BATTLE;
                         else if (ZoneUtils::card_in_zone(state.players[inst->owner].hand, target_id)) from_z = Zone::HAND;
                         else if (ZoneUtils::card_in_zone(state.players[inst->owner].graveyard, target_id)) from_z = Zone::GRAVEYARD;
                         else if (ZoneUtils::card_in_zone(state.players[inst->owner].shield_zone, target_id)) from_z = Zone::SHIELD;

                         TransitionCommand real_trans(target_id, from_z, Zone::MANA, inst->owner);
                         real_trans.execute(state);
                         moved++;
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = moved;
                }
                break;
            }
            case core::CommandType::DESTROY: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int destroyed = 0;
                std::vector<int> destroyed_ids;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         TransitionCommand trans(target_id, Zone::BATTLE, Zone::GRAVEYARD, inst->owner);
                         trans.execute(state);
                         destroyed++;
                         destroyed_ids.push_back(target_id);
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = destroyed;
                    // Store card IDs as comma-separated list in a companion key
                    if (!destroyed_ids.empty()) {
                        std::string ids_key = cmd.output_value_key + "_ids";
                        // Encode as comma-separated integers (abuse int map by storing count at base key)
                        // Better: we can only store one int per key, so we'll store IDs in sequential keys
                        for (size_t i = 0; i < destroyed_ids.size(); ++i) {
                            execution_context[ids_key + "_" + std::to_string(i)] = destroyed_ids[i];
                        }
                        execution_context[ids_key + "_count"] = static_cast<int>(destroyed_ids.size());
                    }
                }
                break;
            }
            case core::CommandType::DISCARD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                int discarded = 0;
                std::vector<int> discarded_ids;
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                        TransitionCommand trans(target_id, Zone::HAND, Zone::GRAVEYARD, inst->owner);
                        trans.execute(state);
                        discarded++;
                        discarded_ids.push_back(target_id);
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = discarded;
                    // Store card IDs in sequential keys
                    if (!discarded_ids.empty()) {
                        std::string ids_key = cmd.output_value_key + "_ids";
                        for (size_t i = 0; i < discarded_ids.size(); ++i) {
                            execution_context[ids_key + "_" + std::to_string(i)] = discarded_ids[i];
                        }
                        execution_context[ids_key + "_count"] = static_cast<int>(discarded_ids.size());
                    }
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
                std::vector<int> returned_ids;
                Zone from_z = parse_zone_string(cmd.from_zone);

                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         Zone actual_from = from_z;
                         // Auto-detect if GRAVEYARD (parse default) but user string wasn't explicitly GRAVEYARD
                         // This implies input was empty or invalid, triggering auto-scan
                         if (actual_from == Zone::GRAVEYARD && cmd.from_zone != "GRAVEYARD") {
                             if (ZoneUtils::card_in_zone(state.players[inst->owner].battle_zone, target_id)) actual_from = Zone::BATTLE;
                             else if (ZoneUtils::card_in_zone(state.players[inst->owner].mana_zone, target_id)) actual_from = Zone::MANA;
                             else if (ZoneUtils::card_in_zone(state.players[inst->owner].shield_zone, target_id)) actual_from = Zone::SHIELD;
                             else if (ZoneUtils::card_in_zone(state.players[inst->owner].hand, target_id)) actual_from = Zone::HAND;
                             else if (ZoneUtils::card_in_zone(state.players[inst->owner].graveyard, target_id)) actual_from = Zone::GRAVEYARD;
                         }

                         TransitionCommand trans(target_id, actual_from, Zone::HAND, inst->owner);
                         trans.execute(state);
                         returned++;
                         returned_ids.push_back(target_id);
                    }
                }
                if (!cmd.output_value_key.empty()) {
                    execution_context[cmd.output_value_key] = returned;
                    // Store card IDs in sequential keys
                    if (!returned_ids.empty()) {
                        std::string ids_key = cmd.output_value_key + "_ids";
                        for (size_t i = 0; i < returned_ids.size(); ++i) {
                            execution_context[ids_key + "_" + std::to_string(i)] = returned_ids[i];
                        }
                        execution_context[ids_key + "_count"] = static_cast<int>(returned_ids.size());
                    }
                }
                break;
            }
            case core::CommandType::BREAK_SHIELD: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id, execution_context);
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         TransitionCommand trans(target_id, Zone::SHIELD, Zone::HAND, inst->owner);
                         trans.execute(state);
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
        return true;
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
                // If filter has explicit count, we might still want AI selection, 
                // but usually filter.count implies "first N" or "random N" in some contexts.
                // For now, let's allow AI selection if DecisionMaker is present.
                
                if (state.decision_maker) {
                     // Create a temporary CommandDef for the selection if needed, or pass current cmd
                     // But wait, filter.count is a hard limit on the *filter*, not necessarily the command's operation amount.
                     // The logic below for command amount is more important.
                }
                
                // Fallback to naive truncation if no AI decision
                targets.resize(n);
            }
        }
        
        // AI Selection for Command Amount
        // If the command specifies an amount (e.g. Discard 2), and we have more targets (e.g. Hand 5),
        // we should let the AI decide which ones to pick.
        int amount = cmd.amount; 
        // Note: resolving amount here is tricky because resolve_targets is called before/during execution.
        // But we passed execution_context, so we can try resolving it.
        // However, resolve_targets doesn't call resolve_amount currently inside itself except for specific logic.
        // In execute_primitive/macro, resolve_amount is called. 
        // We should probably rely on the caller to handle selection?
        // BUT, `execute_primitive` takes `targets` from `resolve_targets` and iterates them all.
        // So `resolve_targets` MUST return only the selected ones if we want to enforce the limit.
        
        // Let's resolve amount again for selection logic
        if (!cmd.input_value_key.empty()) {
             auto it = execution_context.find(cmd.input_value_key);
             if (it != execution_context.end()) amount = it->second;
        }

        if (amount > 0 && targets.size() > (size_t)amount) {
            if (state.decision_maker) {
                targets = state.decision_maker->select_targets(state, cmd, targets, amount);
            } else {
                 // Naive fallback: take first N (or random if we implemented shuffle)
                 // This matches previous behavior
                 targets.resize(amount);
            }
        }

        return targets;
    }

}
