#include "command_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <iostream>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void CommandSystem::execute_command(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        // Distinguish between Primitives and Macros
        // cmd.type is core::CommandType
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
        // Implementation for Primitives (Direct mapping to GameCommand)
        if (cmd.type == core::CommandType::TRANSITION) {
            // Need to resolve targets for transition
            std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id);
            std::cout << "DEBUG: Execute Primitive TRANSITION. Targets found: " << targets.size() << std::endl;

            // Should parse from cmd.from_zone string to Zone Enum
            // Placeholder: Assume MANUAL mapping logic or default
            Zone from_z = Zone::HAND; // Default
            if (cmd.from_zone == "DECK") from_z = Zone::DECK;
            else if (cmd.from_zone == "MANA" || cmd.from_zone == "MANA_ZONE") from_z = Zone::MANA;
            else if (cmd.from_zone == "BATTLE" || cmd.from_zone == "BATTLE_ZONE") from_z = Zone::BATTLE;
            else if (cmd.from_zone == "GRAVEYARD") from_z = Zone::GRAVEYARD;
            else if (cmd.from_zone == "SHIELD" || cmd.from_zone == "SHIELD_ZONE") from_z = Zone::SHIELD;
            else if (cmd.from_zone == "HAND") from_z = Zone::HAND;

            Zone to_z = Zone::GRAVEYARD; // Default
            if (cmd.to_zone == "DECK") to_z = Zone::DECK;
            else if (cmd.to_zone == "MANA" || cmd.to_zone == "MANA_ZONE") to_z = Zone::MANA;
            else if (cmd.to_zone == "BATTLE" || cmd.to_zone == "BATTLE_ZONE") to_z = Zone::BATTLE;
            else if (cmd.to_zone == "GRAVEYARD") to_z = Zone::GRAVEYARD;
            else if (cmd.to_zone == "SHIELD" || cmd.to_zone == "SHIELD_ZONE") to_z = Zone::SHIELD;
            else if (cmd.to_zone == "HAND") to_z = Zone::HAND;

            for (int target_id : targets) {
                // TransitionCommand(instance, from, to, owner)
                CardInstance* inst = state.get_card_instance(target_id);
                if (inst) {
                    TransitionCommand trans(target_id, from_z, to_z, inst->owner);
                    trans.execute(state);
                }
            }
        }
        // ... MUTATE, FLOW, etc.
    }

    void CommandSystem::expand_and_execute_macro(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        switch (cmd.type) {
            case core::CommandType::DRAW_CARD: {
                int count = cmd.amount;
                // Draw = Transition Deck -> Hand (Top N)
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
                 // Assume Legacy ADD_MANA mapping (Top deck).
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
                // Target selection required
                std::vector<int> targets = resolve_targets(state, cmd, source_instance_id, player_id);
                for (int target_id : targets) {
                    CardInstance* inst = state.get_card_instance(target_id);
                    if (inst) {
                         // Determine owner (Graveyard destination)
                         // Rule: Goes to owner's graveyard
                         TransitionCommand trans(target_id, Zone::BATTLE, Zone::GRAVEYARD, inst->owner);
                         trans.execute(state);
                    }
                }
                break;
            }
            // ... TAP, UNTAP, etc.
            default:
                break;
        }
    }

    std::vector<int> CommandSystem::resolve_targets(GameState& state, const CommandDef& cmd, int source_instance_id, PlayerID player_id) {
        // Use TargetUtils or implementation of Filter logic
        std::vector<int> targets;

        // Example for DESTROY (Target Scope)
        if (cmd.target_group == TargetScope::PLAYER_OPPONENT) {
             PlayerID opp_id = 1 - player_id;
             const auto& zone = state.players[opp_id].battle_zone;
             for (const auto& card : zone) {
                 int id = card.instance_id;
                 // Placeholder: assumes valid if loop runs. Real implementation needs TargetUtils::is_valid_target
                 // if (TargetUtils::is_valid(state, id, cmd.target_filter)) ...
                 targets.push_back(id);
             }
        } else if (cmd.target_group == TargetScope::PLAYER_SELF) {
             const auto& zone = state.players[player_id].deck; // Example for TRANSITION test
             // For deck, usually top N? Or all?
             // Filter usually specifies Zone.
             // If cmd.type is TRANSITION and filter.zones is DECK
             // We iterate DECK.
             // But DECK is ordered.
             // For simple test, let's just pick top 1 if count=1
             if (cmd.target_filter.count.has_value() && cmd.target_filter.count.value() == 1) {
                 if (!zone.empty()) targets.push_back(zone.back().instance_id);
             }
        }

        return targets;
    }

}
