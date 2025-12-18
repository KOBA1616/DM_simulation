#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/card/effect_system.hpp" // Added include for EffectSystem
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void GameLogicSystem::handle_play_card(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                           const std::map<core::CardID, core::CardDefinition>& card_db) {
        int card_id = exec.resolve_int(inst.args.value("card", 0));
        int instance_id = card_id; // "card" arg is assumed to be instance_id

        CardInstance* card = state.get_card_instance(instance_id);
        if (!card) return;

        bool is_evolution = false;
        FilterDef evo_filter;
        if (card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            if (def.keywords.evolution) {
                is_evolution = true;
                // Task B: Refined Evolution Filters
                evo_filter.zones = {"BATTLE_ZONE"};
                evo_filter.races = def.races; // Evolution matches race

                // Specific evolution conditions (NEO, etc.) would be handled here by inspecting CardDefinition
                // or a specific evolution field if added.
                // For strict filtering, TargetUtils::is_valid_target uses races/civs if set.
                // Since we set races here, it enforces race matching.

                // Also enforce owner
                evo_filter.owner = "SELF";
            }
        }

        if (is_evolution) {
            std::string selection_key = "$evo_target";

            ContextValue val = exec.get_context_var(selection_key);
            std::vector<int> targets;
            if (std::holds_alternative<std::vector<int>>(val)) {
                targets = std::get<std::vector<int>>(val);
            }

            if (targets.empty()) {
                exec.execution_paused = true;
                exec.waiting_for_key = selection_key;
                state.waiting_for_user_input = true;

                // Populate valid targets for UI using TargetUtils
                // We can reuse PipelineExecutor::handle_select logic by creating a dummy instruction?
                // Or just manual query.
                // Reusing handle_select logic is better but we are inside handle_play_card.

                // Manual query population:
                std::vector<int> valid_targets;
                // Iterate battle zone
                const auto& battle = state.players[state.active_player_id].battle_zone;
                for (const auto& c : battle) {
                    if (card_db.count(c.card_id)) {
                        const auto& def = card_db.at(c.card_id);
                         if (TargetUtils::is_valid_target(c, def, evo_filter, state, state.active_player_id, state.active_player_id)) {
                             valid_targets.push_back(c.instance_id);
                         }
                    }
                }

                state.pending_query = GameState::QueryContext{
                    0, "SELECT_TARGET", {{"min", 1}, {"max", 1}}, valid_targets, {}
                };
                return;
            }

            int base_id = targets[0];
            auto cmd = std::make_unique<AttachCommand>(instance_id, base_id, Zone::HAND);
            state.execute_command(std::move(cmd));

        } else {
             const auto& def = card_db.at(card->card_id);
             Zone dest = Zone::BATTLE;
             if (def.type == CardType::SPELL) dest = Zone::GRAVEYARD;

             auto cmd = std::make_unique<TransitionCommand>(instance_id, Zone::HAND, dest, state.active_player_id);
             state.execute_command(std::move(cmd));
        }
    }

    void GameLogicSystem::handle_resolve_play(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
         // Process On-Play effects
    }

    void GameLogicSystem::handle_attack(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                        const std::map<core::CardID, core::CardDefinition>& card_db) {
         int instance_id = exec.resolve_int(inst.args.value("source", 0));
         auto cmd = std::make_unique<MutateCommand>(instance_id, MutateCommand::MutationType::TAP);
         state.execute_command(std::move(cmd));
    }

    void GameLogicSystem::handle_block(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        // ...
    }

    void GameLogicSystem::handle_resolve_battle(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Compare powers, destroy loser
    }

    void GameLogicSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                              const std::map<core::CardID, core::CardDefinition>& card_db) {
        int shield_id = exec.resolve_int(inst.args.value("shield", -1));
        if (shield_id == -1) return;

        auto cmd = std::make_unique<TransitionCommand>(shield_id, Zone::SHIELD, Zone::HAND, state.active_player_id);
        state.execute_command(std::move(cmd));

        const auto* card = state.get_card_instance(shield_id);
        if (card && card_db.count(card->card_id)) {
            const auto& def = card_db.at(card->card_id);
            if (def.keywords.shield_trigger) {

                std::string decision_key = "$strigger_decision_" + std::to_string(shield_id);
                ContextValue val = exec.get_context_var(decision_key);

                bool use_trigger = false;
                bool decision_made = false;

                if (std::holds_alternative<int>(val)) {
                    use_trigger = std::get<int>(val) == 1;
                    decision_made = true;
                } else if (std::holds_alternative<std::vector<int>>(val)) {
                     const auto& vec = std::get<std::vector<int>>(val);
                     if(!vec.empty()) use_trigger = vec[0] == 1;
                     decision_made = true;
                }

                if (!decision_made) {
                    std::cout << "[GameLogic] S-Trigger detected: " << def.name << ". Pausing for input." << std::endl;
                    exec.waiting_for_key = decision_key;
                    exec.execution_paused = true;

                    state.waiting_for_user_input = true;
                    state.pending_query = GameState::QueryContext{
                        0, "SELECT_OPTION", {}, {}, {"No", "Yes"} // 0=No, 1=Yes
                    };
                    return;
                }

                if (use_trigger) {
                    std::cout << "[GameLogic] Activating S-Trigger: " << def.name << std::endl;

                    // Task A: Complete Effect Resolution
                    // 1. Play (Free)
                    Zone dest = (def.type == CardType::SPELL) ? Zone::GRAVEYARD : Zone::BATTLE;
                    auto play_cmd = std::make_unique<TransitionCommand>(shield_id, Zone::HAND, dest, state.active_player_id);
                    state.execute_command(std::move(play_cmd));

                    // 2. Compile Effects
                    std::vector<Instruction> compiled_effects;
                    std::map<std::string, int> ctx; // Empty initial context

                    for (const auto& eff : def.effects) {
                         // Check trigger? S-Trigger execution usually executes ALL main effects (Spells)
                         // Or CIP effects (Creatures).
                         if (def.type == CardType::SPELL) {
                             EffectSystem::instance().compile_effect(state, eff, shield_id, ctx, card_db, compiled_effects);
                         } else {
                             // For creatures, playing it triggers ON_PLAY normally via `resolve_trigger`.
                             // However, usually S-Trigger creatures have CIP effects.
                             // The `TransitionCommand` moves it to Battle Zone.
                             // The GameLoop or TriggerSystem should pick up the ON_PLAY trigger.
                             // BUT, we are inside a pipeline execution (BREAK_SHIELD).
                             // We might need to ensure triggers are processed.
                             // Since we are handling this inline, we assume standard trigger system handles CIP.
                             // So we only compile effects for SPELLS here.
                         }
                    }

                    if (!compiled_effects.empty()) {
                         // Inject instructions into the pipeline
                         auto block = std::make_shared<std::vector<Instruction>>(compiled_effects);
                         exec.call_stack.push_back({block, 0, LoopContext{}});
                    }
                }
            }
        }
    }

    void GameLogicSystem::handle_mana_charge(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
         // ...
    }

    void GameLogicSystem::handle_resolve_reaction(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                                  const std::map<core::CardID, core::CardDefinition>& card_db) {
         // ...
    }

    void GameLogicSystem::handle_use_ability(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                             const std::map<core::CardID, core::CardDefinition>& card_db) {
         // ...
    }

    void GameLogicSystem::handle_select_target(PipelineExecutor& exec, GameState& state, const Instruction& inst) {
        exec.execution_paused = true;
        // ... set query ...
    }

}
