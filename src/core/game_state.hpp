#pragma once
#include "types.hpp"
#include "constants.hpp"
#include <vector>
#include <array>
#include <random>
#include <stdexcept>

namespace dm::core {

    struct CardInstance {
        CardID card_id; // The definition ID
        int instance_id; // Unique ID for this instance during the game
        bool is_tapped = false;
        bool summoning_sickness = false;
        bool is_face_down = false; // For shields or terror pit etc? Shields are face down by default.
        
        // Constructors
        CardInstance() : card_id(0), instance_id(-1) {}
        CardInstance(CardID cid, int iid) : card_id(cid), instance_id(iid) {}
    };

    struct Player {
        PlayerID id;
        std::vector<CardInstance> hand;
        std::vector<CardInstance> mana_zone;
        std::vector<CardInstance> battle_zone;
        std::vector<CardInstance> graveyard;
        std::vector<CardInstance> shield_zone;
        std::vector<CardInstance> deck;
        std::vector<CardInstance> hyper_spatial_zone;
        std::vector<CardInstance> gr_deck;
    };

    struct GameState {
        int turn_number = 1;
        PlayerID active_player_id = 0;
        std::array<Player, 2> players;
        
        // Determinism: std::mt19937 inside State [Q69]
        std::mt19937 rng;

        GameState(uint32_t seed) : rng(seed) {
            players[0].id = 0;
            players[1].id = 1;
        }

        Player& get_active_player() {
            return players[active_player_id];
        }

        Player& get_non_active_player() {
            return players[1 - active_player_id];
        }
        
        // Error Handling [Q43, Q87]
        void panic(const char* message) const {
            throw std::runtime_error(message);
        }
    };
}
