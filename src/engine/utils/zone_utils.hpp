#pragma once
#include "core/game_state.hpp"
#include <vector>
#include <algorithm>
#include <optional>

namespace dm::engine {
    class ZoneUtils {
    public:
        // Helper to remove card from a specific zone by instance_id.
        // Returns true if found and removed.
        static bool remove_card(std::vector<dm::core::CardInstance>& zone, int instance_id) {
            auto it = std::find_if(zone.begin(), zone.end(), [&](const dm::core::CardInstance& c){ return c.instance_id == instance_id; });
            if (it != zone.end()) {
                zone.erase(it);
                return true;
            }
            return false;
        }

        // Handle cleanup when a card leaves the battle zone.
        static void on_leave_battle_zone(dm::core::GameState& game_state, dm::core::CardInstance& card) {
            using namespace dm::core;
            if (card.underlying_cards.empty()) return;

            PlayerID owner_id = 0;
            // Phase A: Use O(1) owner map if available, or infer
            if (card.instance_id >= 0 && card.instance_id < (int)game_state.card_owner_map.size()) {
                owner_id = game_state.card_owner_map[card.instance_id];
            } else {
                // Fallback implies risk, but acceptable for now
                // Assuming card.owner field not available yet
            }

            Player& owner = game_state.players[owner_id];

            for (auto& under : card.underlying_cards) {
                under.is_tapped = false;
                under.power_mod = 0;
                under.underlying_cards.clear();
                owner.graveyard.push_back(under);
            }
            card.underlying_cards.clear();
        }

        // Helper to find and remove a card from ANY zone of ANY player (or Buffer).
        // Returns the card instance if found, and updates the game state (removes it).
        // Also triggers on_leave_battle_zone if removed from Battle Zone.
        static std::optional<dm::core::CardInstance> find_and_remove(dm::core::GameState& game_state, int instance_id) {
            using namespace dm::core;

            // 1. Check Effect Buffer
            auto& buffer = game_state.effect_buffer;
            auto it_buf = std::find_if(buffer.begin(), buffer.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
            if (it_buf != buffer.end()) {
                CardInstance c = *it_buf;
                buffer.erase(it_buf);
                return c;
            }

            // 2. Check Players
            for (auto& p : game_state.players) {
                // Battle Zone
                auto it_bz = std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (it_bz != p.battle_zone.end()) {
                    CardInstance c = *it_bz;
                    on_leave_battle_zone(game_state, *it_bz); // Cleanup hierarchy BEFORE erasing
                    c = *it_bz; // Copy modified state (empty underlying)
                    p.battle_zone.erase(it_bz);
                    return c;
                }

                // Hand
                auto it_hand = std::find_if(p.hand.begin(), p.hand.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (it_hand != p.hand.end()) {
                    CardInstance c = *it_hand;
                    p.hand.erase(it_hand);
                    return c;
                }

                // Mana
                auto it_mana = std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (it_mana != p.mana_zone.end()) {
                    CardInstance c = *it_mana;
                    p.mana_zone.erase(it_mana);
                    return c;
                }

                // Shield
                auto it_shield = std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (it_shield != p.shield_zone.end()) {
                    CardInstance c = *it_shield;
                    p.shield_zone.erase(it_shield);
                    return c;
                }

                // Graveyard
                auto it_grave = std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (it_grave != p.graveyard.end()) {
                    CardInstance c = *it_grave;
                    p.graveyard.erase(it_grave);
                    return c;
                }
            }

            return std::nullopt;
        }
    };
}
