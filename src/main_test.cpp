#include "utils/csv_loader.hpp"
#include "core/game_state.hpp"
#include "engine/flow/phase_manager.hpp"
#include "engine/action_gen/action_generator.hpp"
#include "engine/effects/effect_resolver.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <filesystem>

using namespace dm::core;
using namespace dm::engine;
using namespace dm::utils;

// Helper to print game state
void print_state(const GameState& state) {
    std::cout << "Turn: " << state.turn_number << " | Phase: " << (int)state.current_phase 
              << " | Active: P" << (int)state.active_player_id << "\n";
    std::cout << "P0 Hand: " << state.players[0].hand.size() << " Mana: " << state.players[0].mana_zone.size() 
              << " Battle: " << state.players[0].battle_zone.size() << " Shields: " << state.players[0].shield_zone.size() << "\n";
    std::cout << "P1 Hand: " << state.players[1].hand.size() << " Mana: " << state.players[1].mana_zone.size() 
              << " Battle: " << state.players[1].battle_zone.size() << " Shields: " << state.players[1].shield_zone.size() << "\n";
    std::cout << "--------------------------------------------------\n";
}

int main() {
    try {
        // 1. Load Cards
        auto card_db = CsvLoader::load_cards("data/cards.csv");
        std::cout << "Loaded " << card_db.size() << " cards.\n";

        // 2. Initialize Game
        uint32_t seed = static_cast<uint32_t>(std::chrono::system_clock::now().time_since_epoch().count());
        GameState game_state(seed);

        // Create dummy decks
        for (int i = 0; i < 40; ++i) {
            // Add random cards from DB (ID 1 to 5)
            CardID cid = (i % 5) + 1;
            game_state.players[0].deck.emplace_back(cid, i);
            game_state.players[1].deck.emplace_back(cid, i + 100);
        }

        // If historical stats JSON exists, load and compute initial deck sums
        std::string stats_path = "data/card_stats.json";
        if (std::filesystem::exists(stats_path)) {
            bool ok = game_state.load_card_stats_from_json(stats_path);
            std::cout << "Loaded historical stats: " << (ok ? "ok" : "fail") << "\n";
            // Build deck list for player 0 and compute initial sums
            std::vector<CardID> deck_list;
            for (const auto &ci : game_state.players[0].deck) deck_list.push_back(ci.card_id);
            game_state.compute_initial_deck_sums(deck_list);
            auto pot = game_state.get_library_potential();
            std::cout << "Initial library potential (first 8 dims): ";
            for (int i = 0; i < 8; ++i) std::cout << pot[i] << " ";
            std::cout << "\n";
        }

        PhaseManager::start_game(game_state, card_db);

        // 3. Game Loop
        int steps = 0;
        GameResult result = GameResult::NONE;

        while (steps < 1000) { // Safety break
            steps++;
            
            if (PhaseManager::check_game_over(game_state, result)) {
                std::cout << "Game Over! Result: " << (int)result << "\n";
                break;
            }

            // Generate Actions
            auto actions = ActionGenerator::generate_legal_actions(game_state, card_db);
            
            if (actions.empty()) {
                // Should not happen if PASS is always available in relevant phases
                // But Start/Draw/End phases might not have actions, just auto-transition
                PhaseManager::next_phase(game_state, card_db);
                continue;
            }

            // Pick Random Action
            std::uniform_int_distribution<size_t> dist(0, actions.size() - 1);
            const Action& action = actions[dist(game_state.rng)];

            // std::cout << "Action: " << action.to_string() << "\n";

            // Resolve
            EffectResolver::resolve_action(game_state, action, card_db);

            // If PASS, move phase
            if (action.type == ActionType::PASS) {
                PhaseManager::next_phase(game_state, card_db);
            }
            
            // print_state(game_state);
        }
        
        print_state(game_state);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
