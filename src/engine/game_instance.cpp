#include "game_instance.hpp"
#include "systems/flow/phase_manager.hpp"
#include "engine/game_command/game_command.hpp"
#include "engine/game_command/action_commands.hpp" // Added
#include "engine/game_command/commands.hpp" // Added for DeclareReactionCommand
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/continuous_effect_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/card/effect_system.hpp" // Added for EffectSystem
#include <functional>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;
    using namespace dm::engine::game_command; // Added

    GameInstance::GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db)
        : state(seed), card_db(db) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Refactored: Use TriggerManager to setup event handling logic
        systems::TriggerManager::setup_event_handling(state, trigger_manager, card_db);
    }

    GameInstance::GameInstance(uint32_t seed)
        : state(seed), card_db(CardRegistry::get_all_definitions_ptr()) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Refactored: Use TriggerManager to setup event handling logic
        systems::TriggerManager::setup_event_handling(state, trigger_manager, card_db);
    }

    void GameInstance::start_game() {
        if (card_db) {
            PhaseManager::start_game(state, *card_db);
        }
    }

    void GameInstance::resolve_action(const core::Action& action) {
        if (!pipeline) {
            std::cerr << "FATAL: pipeline is null!" << std::endl;
            return;
        }
        if (!card_db) {
             std::cerr << "FATAL: card_db is null!" << std::endl;
             return;
        }
        state.active_pipeline = pipeline;

        // --- Migration to GameCommand ---
        // Instead of dispatching directly via GameLogicSystem, we convert to Command if possible
        // to ensure Undo/Redo consistency (treating Action as a single reversible unit).

        std::unique_ptr<GameCommand> cmd = nullptr;

        switch (action.type) {
            case PlayerIntent::PLAY_CARD:
            case PlayerIntent::PLAY_CARD_INTERNAL:
                {
                    auto p_cmd = std::make_unique<PlayCardCommand>(action.source_instance_id);
                    p_cmd->is_spell_side = action.is_spell_side;
                    p_cmd->spawn_source = action.spawn_source;
                    p_cmd->target_slot_index = action.target_slot_index;
                    cmd = std::move(p_cmd);
                }
                break;
            case PlayerIntent::ATTACK_CREATURE:
                cmd = std::make_unique<AttackCommand>(action.source_instance_id, action.target_instance_id);
                break;
            case PlayerIntent::ATTACK_PLAYER:
                cmd = std::make_unique<AttackCommand>(action.source_instance_id, -1, action.target_player);
                break;
            case PlayerIntent::BLOCK:
                cmd = std::make_unique<BlockCommand>(action.source_instance_id);
                break;
            case PlayerIntent::MANA_CHARGE:
                cmd = std::make_unique<ManaChargeCommand>(action.source_instance_id);
                break;
            case PlayerIntent::PASS:
                cmd = std::make_unique<PassCommand>();
                break;
            case PlayerIntent::USE_ABILITY:
                cmd = std::make_unique<UseAbilityCommand>(action.source_instance_id, action.target_instance_id);
                break;
            case PlayerIntent::DECLARE_REACTION:
                {
                   int idx = action.slot_index;
                   bool is_pass = (idx == -1);
                   cmd = std::make_unique<DeclareReactionCommand>(state.active_player_id, is_pass, idx);
                }
                break;
            case PlayerIntent::SELECT_OPTION:
                {
                    // Handle option selection for pending effects
                    int effect_idx = action.slot_index;
                    int option_idx = action.target_slot_index;
                    
                    if (effect_idx >= 0 && effect_idx < (int)state.pending_effects.size()) {
                        auto& pe = state.pending_effects[effect_idx];
                        
                        // TODO: Implement option handling if needed in the future
                        // Currently, options are handled via CommandDef in effect_def.commands
                        
                        // Remove the pending effect
                        state.pending_effects.erase(state.pending_effects.begin() + effect_idx);
                    }
                }
                break;
            case PlayerIntent::SELECT_NUMBER:
                {
                    // Handle number selection for pending effects
                    int effect_idx = action.slot_index;
                    int chosen_number = action.target_instance_id; // The chosen number is stored in target_instance_id
                    
                    if (effect_idx >= 0 && effect_idx < (int)state.pending_effects.size()) {
                        auto& pe = state.pending_effects[effect_idx];
                        
                        // Store the chosen number in execution_context if output_value_key is specified
                        if (pe.effect_def.has_value() && !pe.effect_def->condition.str_val.empty()) {
                            // The output key is stored in effect_def.condition.str_val (as per SelectNumberHandler)
                            std::string output_key = pe.effect_def->condition.str_val;
                            pe.execution_context[output_key] = chosen_number;
                        }
                        
                        // Execute continuation actions with the updated context
                        if (pe.effect_def.has_value()) {
                            for (const auto& act : pe.effect_def->actions) {
                                EffectSystem::instance().resolve_action(state, act, pe.source_instance_id, pe.execution_context, *card_db);
                            }
                        }
                        
                        // Remove the pending effect
                        state.pending_effects.erase(state.pending_effects.begin() + effect_idx);
                    }
                }
                break;
            default:
                // Fallback for atomic actions (PAY_COST, RESOLVE_PLAY) or legacy
                systems::GameLogicSystem::dispatch_action(*pipeline, state, action, *card_db);
                systems::ContinuousEffectSystem::recalculate(state, *card_db);
                return;
        }

        if (cmd) {
            state.execute_command(std::move(cmd));
        } else {
             // If cmd was not created (e.g. unmapped intent), fallback
             systems::GameLogicSystem::dispatch_action(*pipeline, state, action, *card_db);
        }

        systems::ContinuousEffectSystem::recalculate(state, *card_db);
    }

    void GameInstance::undo() {
        if (state.command_history.empty()) return;

        auto& cmd = state.command_history.back();
        cmd->invert(state);
        state.command_history.pop_back();
    }

    // ... [rest of file unchanged]

    void GameInstance::initialize_card_stats(int deck_size) {
        if (card_db) {
            state.initialize_card_stats(*card_db, deck_size);
        }
    }

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
        // [Existing implementation remains unchanged]
        // 1. Reset Game State
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
        state.winner = GameResult::NONE;
        state.pending_effects.clear();
        state.current_attack = AttackState(); // Reset attack context

        // Clear all zones for both players
        for (auto& p : state.players) {
            p.hand.clear();
            p.battle_zone.clear();
            p.mana_zone.clear();
            p.graveyard.clear();
            p.shield_zone.clear();
            p.deck.clear();
        }

        // Instance ID counter
        int instance_id_counter = 0;

        // Fill decks
        // Player 0 (Me)
        if (!config.my_deck.empty()) {
             for (int cid : config.my_deck) {
                  state.players[0].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)0);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[0].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)0);
             }
        }

        // Player 1 (Enemy)
        if (!config.enemy_deck.empty()) {
             for (int cid : config.enemy_deck) {
                  state.players[1].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)1);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[1].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)1);
             }
        }

        // 2. Setup My Resources (Player 0)
        Player& me = state.players[0];

        // Hand
        for (int cid : config.my_hand_cards) {
            me.hand.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // Battle Zone
        for (int cid : config.my_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.summoning_sickness = false; // Assume creatures on board are ready
            me.battle_zone.push_back(c);
        }

        // Mana Zone
        for (int cid : config.my_mana_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.is_tapped = false;
            me.mana_zone.push_back(c);
        }

        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
                me.mana_zone.emplace_back(1, instance_id_counter++, me.id);
            }
        }

        // Graveyard
        for (int cid : config.my_grave_yard) {
            me.graveyard.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // My Shields (Player 0)
        for (int cid : config.my_shields) {
             me.shield_zone.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // 3. Setup Enemy Resources (Player 1)
        Player& enemy = state.players[1];

        // Enemy Battle Zone
        for (int cid : config.enemy_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, enemy.id);
            c.summoning_sickness = false;
            enemy.battle_zone.push_back(c);
        }

        // Enemy Shields
        for (int i = 0; i < config.enemy_shield_count; ++i) {
             enemy.shield_zone.emplace_back(1, instance_id_counter++, enemy.id);
        }

        // Initialize Owner Map [Phase A]
        state.card_owner_map.resize(instance_id_counter);

        auto populate_owner = [&](const Player& p) {
            auto register_cards = [&](const std::vector<CardInstance>& cards) {
                for (const auto& c : cards) {
                    if (c.instance_id >= 0 && c.instance_id < (int)state.card_owner_map.size()) {
                        state.card_owner_map[c.instance_id] = p.id;
                    }
                }
            };
            register_cards(p.hand);
            register_cards(p.battle_zone);
            register_cards(p.mana_zone);
            register_cards(p.graveyard);
            register_cards(p.shield_zone);
            register_cards(p.deck);
            register_cards(p.hyper_spatial_zone);
            register_cards(p.gr_deck);
        };

        populate_owner(state.players[0]);
        populate_owner(state.players[1]);
    }

    // A. Interactive Processing implementation
    void GameInstance::resume_processing(const std::vector<int>& inputs) {
         if (pipeline && pipeline->execution_paused) {
             // Basic support for single integer input for now (common case)
             dm::engine::systems::ContextValue val;
             if (inputs.empty()) {
                  // No input?
             } else if (inputs.size() == 1) {
                  val = inputs[0];
             } else {
                  val = inputs;
             }
             pipeline->resume(state, *card_db, val);
             systems::ContinuousEffectSystem::recalculate(state, *card_db);
         }
    }

}
