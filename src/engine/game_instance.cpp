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

    GameInstance::GameInstance(uint32_t seed, std::shared_ptr<const std::map<core::CardID, core::CardDefinition>> db)
        : state(seed), card_db(db) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Wire up GameState's event dispatcher to TriggerManager
        state.event_dispatcher = [this](const core::GameEvent& event) {
            trigger_manager->dispatch(event, state);
            if (card_db) {
                trigger_manager->check_triggers(event, state, *card_db);
                trigger_manager->check_reactions(event, state, *card_db);
            }
        };
    }

    GameInstance::GameInstance(uint32_t seed)
        : state(seed), card_db(CardRegistry::get_all_definitions_ptr()) {
        trigger_manager = std::make_shared<systems::TriggerManager>();
        pipeline = std::make_shared<systems::PipelineExecutor>();

        // Wire up GameState's event dispatcher to TriggerManager
        state.event_dispatcher = [this](const core::GameEvent& event) {
            trigger_manager->dispatch(event, state);
            if (card_db) {
                trigger_manager->check_triggers(event, state, *card_db);
                trigger_manager->check_reactions(event, state, *card_db);
            }
        };
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
        systems::GameLogicSystem::dispatch_action(*pipeline, state, action, *card_db);
        systems::ContinuousEffectSystem::recalculate(state, *card_db);
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
        if (card_db) {
            state.initialize_card_stats(*card_db, deck_size);
        }
    }

    void GameInstance::reset_with_scenario(const ScenarioConfig& config) {
        // 1. Reset Game State
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
        state.winner = GameResult::NONE;
        state.pending_effects.clear();
        state.reaction_stack.clear();
        state.current_attack = AttackState(); // Reset attack context
        state.command_history.clear(); // Clear history for fresh start

        // Clear all zones for both players
        for (auto& p : state.players) {
            p.hand.clear();
            p.battle_zone.clear();
            p.mana_zone.clear();
            p.graveyard.clear();
            p.shield_zone.clear();
            p.deck.clear();
            p.stack.clear();
            p.effect_buffer.clear();
            p.hyper_spatial_zone.clear();
            p.gr_deck.clear();
        }

        // Clear owner map
        state.card_owner_map.clear();

        // Instance ID counter
        int instance_id_counter = 0;

        // Helper to add cards
        auto add_cards = [&](const std::vector<int>& cards, Zone zone, PlayerID pid, std::function<void(CardInstance&)> setup = nullptr) {
            for (int cid : cards) {
                CardInstance c((CardID)cid, instance_id_counter++, pid);
                if (setup) setup(c);
                state.add_card_to_zone(c, zone, pid);
            }
        };

        // Fill decks
        // Player 0 (Me)
        if (!config.my_deck.empty()) {
             add_cards(config.my_deck, Zone::DECK, 0);
        } else {
             // Fallback to dummy deck
             std::vector<int> dummy(30, 1);
             add_cards(dummy, Zone::DECK, 0);
        }

        // Player 1 (Enemy)
        if (!config.enemy_deck.empty()) {
             add_cards(config.enemy_deck, Zone::DECK, 1);
        } else {
             // Fallback to dummy deck
             std::vector<int> dummy(30, 1);
             add_cards(dummy, Zone::DECK, 1);
        }

        // 2. Setup My Resources (Player 0)
        PlayerID me_id = state.players[0].id; // 0

        // Hand
        add_cards(config.my_hand_cards, Zone::HAND, me_id);

        // Battle Zone
        add_cards(config.my_battle_zone, Zone::BATTLE, me_id, [](CardInstance& c){
            c.summoning_sickness = false; // Assume creatures on board are ready
        });

        // Mana Zone
        add_cards(config.my_mana_zone, Zone::MANA, me_id, [](CardInstance& c){
            c.is_tapped = false;
        });

        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
                CardInstance c(1, instance_id_counter++, me_id);
                c.is_tapped = false;
                state.add_card_to_zone(c, Zone::MANA, me_id);
            }
        }

        // Graveyard
        add_cards(config.my_grave_yard, Zone::GRAVEYARD, me_id);

        // My Shields (Player 0)
        add_cards(config.my_shields, Zone::SHIELD, me_id);

        // 3. Setup Enemy Resources (Player 1)
        PlayerID enemy_id = state.players[1].id; // 1

        // Enemy Battle Zone
        add_cards(config.enemy_battle_zone, Zone::BATTLE, enemy_id, [](CardInstance& c){
            c.summoning_sickness = false;
        });

        // Enemy Shields
        if (config.enemy_shield_count > 0) {
             std::vector<int> shields(config.enemy_shield_count, 1);
             add_cards(shields, Zone::SHIELD, enemy_id);
        }
    }

}
