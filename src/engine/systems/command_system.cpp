#include "command_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp" // Added
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

    // Helper to ensure evaluators are registered (similar to GenericCardSystem)
    // Note: In a real system, registration should be done at startup in a centralized place.
    // However, ConditionSystem is a singleton, so we need to ensure it's populated.
    // Since generic_card_system.cpp handles registration for now, we might need to rely on it
    // or duplicate registration here if GenericCardSystem hasn't run.
    // A better approach is to expose the registration function or do it in GameInstance.
    // For now, we'll try to trigger registration if needed, but since generic_card_system.cpp
    // uses static local flags, we can't easily trigger it from here without linking or exposing.
    // Hack: We will register them here too if empty. This assumes Evaluators are stateless.

    // Logic for Flow requires evaluators to be registered.
    // For unit tests invoking CommandSystem directly, we need to ensure the registry is populated.
    // We use a static initializer helper to do this safely once.

    // We need definitions of Evaluators. They are currently in condition_system.hpp.
    #include "engine/systems/card/condition_system.hpp"

    static bool _evaluators_ensured = [](){
        ConditionSystem& sys = ConditionSystem::instance();
        // Register core evaluators if not present
        if (!sys.get_evaluator("DURING_YOUR_TURN")) {
            sys.register_evaluator("DURING_YOUR_TURN", std::make_unique<TurnEvaluator>());
            sys.register_evaluator("DURING_OPPONENT_TURN", std::make_unique<TurnEvaluator>());
            sys.register_evaluator("MANA_ARMED", std::make_unique<ManaArmedEvaluator>());
            sys.register_evaluator("SHIELD_COUNT", std::make_unique<ShieldCountEvaluator>());
            sys.register_evaluator("OPPONENT_PLAYED_WITHOUT_MANA", std::make_unique<OpponentPlayedWithoutManaEvaluator>());
            sys.register_evaluator("CIVILIZATION_MATCH", std::make_unique<CivilizationMatchEvaluator>());
            sys.register_evaluator("FIRST_ATTACK", std::make_unique<FirstAttackEvaluator>());
            sys.register_evaluator("COMPARE_STAT", std::make_unique<CompareStatEvaluator>());
            sys.register_evaluator("OPPONENT_DRAW_COUNT", std::make_unique<OpponentCardsDrawnEvaluator>());
        }
        return true;
    }();

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

            // Design Intent: Map string-based JSON commands to internal efficient Enums.
            // Invalid strings are ignored to prevent undefined behavior or unintended effects.
            bool valid_mutation = true;
            MutateCommand::MutationType m_type = MutateCommand::MutationType::TAP; // Default init, but checked via flag

            if (cmd.mutation_kind == "TAP") m_type = MutateCommand::MutationType::TAP;
            else if (cmd.mutation_kind == "UNTAP") m_type = MutateCommand::MutationType::UNTAP;
            else if (cmd.mutation_kind == "POWER_MOD") m_type = MutateCommand::MutationType::POWER_MOD;
            else if (cmd.mutation_kind == "ADD_KEYWORD") m_type = MutateCommand::MutationType::ADD_KEYWORD;
            else if (cmd.mutation_kind == "REMOVE_KEYWORD") m_type = MutateCommand::MutationType::REMOVE_KEYWORD;
            else if (cmd.mutation_kind == "ADD_PASSIVE_EFFECT") m_type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
            else if (cmd.mutation_kind == "ADD_COST_MODIFIER") m_type = MutateCommand::MutationType::ADD_COST_MODIFIER;
            else if (cmd.mutation_kind == "ADD_PENDING_EFFECT") m_type = MutateCommand::MutationType::ADD_PENDING_EFFECT;
            else valid_mutation = false;

            if (valid_mutation) {
                for (int target_id : targets) {
                    MutateCommand mutate(target_id, m_type, cmd.amount, cmd.str_param);
                    mutate.execute(state);
                }
            } else {
                std::cerr << "Warning: Unknown mutation kind: " << cmd.mutation_kind << std::endl;
            }
        } else if (cmd.type == core::CommandType::FLOW) {
            bool condition_met = true;

            // Design Intent: Evaluate the condition using the centralized ConditionSystem.
            // If condition is absent, default to true.
            if (cmd.condition.has_value()) {
                 // Note: We need card_db for some conditions.
                 // Performance: Getting all definitions every check might be heavy if not optimized,
                 // but CardRegistry is a static lookup map.
                 const auto& card_db = CardRegistry::get_all_definitions();
                 // Empty context for now, or we could pass local variables if we implement local scope.
                 std::map<std::string, int> empty_context;

                 condition_met = ConditionSystem::instance().evaluate_def(
                     state, cmd.condition.value(), source_instance_id, card_db, empty_context
                 );
            }

            const auto& commands_to_run = condition_met ? cmd.if_true : cmd.if_false;
            for (const auto& sub_cmd : commands_to_run) {
                execute_command(state, sub_cmd, source_instance_id, player_id);
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
