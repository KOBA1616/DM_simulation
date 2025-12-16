#include "command_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include <iostream>
#include <algorithm>

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

    void CommandSystem::execute_command(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        switch (cmd.type) {
            case core::CommandType::TRANSITION:
            case core::CommandType::MUTATE:
            case core::CommandType::FLOW:
            case core::CommandType::QUERY:
                execute_primitive(state, cmd, source_instance_id, player_id);
                break;
            default:
                expand_and_execute_macro(state, cmd, source_instance_id, player_id);
                break;
        }
    }

    void CommandSystem::execute_primitive(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        if (cmd.type == core::CommandType::TRANSITION) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id);

            Zone from_z = parse_zone_string(cmd.from_zone);
            Zone to_z = parse_zone_string(cmd.to_zone);

            for (int target_id : targets) {
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    TransitionCommand trans(target_id, from_z, to_z, inst->owner);
                    trans.execute(state);
                }
            }
        } else if (cmd.type == core::CommandType::MUTATE) {
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id);
            MutateCommand::MutationType m_type;
            bool valid = true;

            if (cmd.mutation_kind == "TAP") m_type = MutateCommand::MutationType::TAP;
            else if (cmd.mutation_kind == "UNTAP") m_type = MutateCommand::MutationType::UNTAP;
            else if (cmd.mutation_kind == "POWER_MOD") m_type = MutateCommand::MutationType::POWER_MOD;
            else if (cmd.mutation_kind == "ADD_KEYWORD") m_type = MutateCommand::MutationType::ADD_KEYWORD;
            else if (cmd.mutation_kind == "REMOVE_KEYWORD") m_type = MutateCommand::MutationType::REMOVE_KEYWORD;
            else if (cmd.mutation_kind == "ADD_PASSIVE_EFFECT") m_type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
            else if (cmd.mutation_kind == "ADD_COST_MODIFIER") m_type = MutateCommand::MutationType::ADD_COST_MODIFIER;
            else if (cmd.mutation_kind == "ADD_PENDING_EFFECT") m_type = MutateCommand::MutationType::ADD_PENDING_EFFECT;
            else {
                std::cerr << "Warning: Unknown mutation kind '" << cmd.mutation_kind << "'" << std::endl;
                valid = false;
            }

            if (valid) {
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, m_type, cmd.amount, cmd.str_param);
                    mutate.execute(state);
                }
            }
        }
    }

    void CommandSystem::expand_and_execute_macro(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        switch (cmd.type) {
            case core::CommandType::DRAW_CARD: {
                int count = cmd.amount;
                for (int i = 0; i < count; ++i) {
                     const auto& deck = state.players[player_id].deck;
                     if (!deck.empty()) {
                         int card_inst_id = deck.back().instance_id;
                         TransitionCommand trans(card_inst_id, Zone::DECK, Zone::HAND, player_id);
                         trans.execute(state);
                     }
                }
                break;
            }
            case core::CommandType::MANA_CHARGE: {
                 int count = cmd.amount;
                 for (int i = 0; i < count; ++i) {
                     const auto& deck = state.players[player_id].deck;
                     if (!deck.empty()) {
                         int card_inst_id = deck.back().instance_id;
                         TransitionCommand trans(card_inst_id, Zone::DECK, Zone::MANA, player_id);
                         trans.execute(state);
                     }
                }
                 break;
            }
            case core::CommandType::DESTROY: {
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id);
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         TransitionCommand trans(target_id, Zone::BATTLE, Zone::GRAVEYARD, inst->owner);
                         trans.execute(state);
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    std::vector<int> CommandSystem::resolve_targets(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
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
                                filter, state, player_id, pid, false, nullptr)) {
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
                                     card, def, filter, state, player_id, pid, false, nullptr)) {
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
