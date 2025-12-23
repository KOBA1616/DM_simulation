#include "game_instance.hpp"
#include "systems/flow/phase_manager.hpp"
#include "engine/game_command/game_command.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/continuous_effect_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <functional>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    GameInstance::GameInstance(uint32_t seed, const std::map<core::CardID, core::CardDefinition>& db)
        : state(seed), card_db_ptr(std::make_shared<std::map<core::CardID, core::CardDefinition>>(db)), card_db(*card_db_ptr) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Wire up GameState's event dispatcher to TriggerManager
        state.event_dispatcher = [this](const core::GameEvent& event) {
            trigger_manager->dispatch(event, state);
            trigger_manager->check_triggers(event, state, card_db);
            trigger_manager->check_reactions(event, state, card_db);
        };
    }

    GameInstance::GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db)
        : state(seed), card_db_ptr(db), card_db(*card_db_ptr) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Wire up GameState's event dispatcher to TriggerManager
        state.event_dispatcher = [this](const core::GameEvent& event) {
            trigger_manager->dispatch(event, state);
            trigger_manager->check_triggers(event, state, card_db);
            trigger_manager->check_reactions(event, state, card_db);
        };
    }

    GameInstance::GameInstance(uint32_t seed)
        : state(seed), card_db_ptr(CardRegistry::get_all_definitions_ptr()), card_db(*card_db_ptr) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Wire up GameState's event dispatcher to TriggerManager
        state.event_dispatcher = [this](const core::GameEvent& event) {
            trigger_manager->dispatch(event, state);
            trigger_manager->check_triggers(event, state, card_db);
            trigger_manager->check_reactions(event, state, card_db);
        };
    }

    void GameInstance::start_game() {
        PhaseManager::start_game(state, card_db);
    }

    void GameInstance::resolve_action(const core::Action& action) {
        if (!pipeline) {
            std::cerr << "FATAL: pipeline is null!" << std::endl;
            return;
        }
        state.active_pipeline = pipeline;
        systems::GameLogicSystem::dispatch_action(*pipeline, state, action, card_db);
        systems::ContinuousEffectSystem::recalculate(state, card_db);
    }

    void GameInstance::undo() {
        if (state.command_history.empty()) return;

        // Get the last command (ref to shared_ptr)
        // Using auto& to avoid copying shared_ptr, though copy is cheap.
        // It points to shared_ptr<GameCommand>.
        auto& cmd = state.command_history.back();

        // Execute invert logic
        cmd->invert(state);

        // Remove from history
        state.command_history.pop_back();
    }

    void GameInstance::initialize_card_stats(int deck_size) {
        state.initialize_card_stats(card_db, deck_size);
    }

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
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

        // Populate owner map
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

}
