#include "game_state.hpp"
#include <functional>

namespace dm::core {

    // Simple hash combine function
    template <class T>
    inline void hash_combine(std::size_t& seed, const T& v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }

    uint64_t GameState::calculate_hash() const {
        std::size_t seed = 0;

        // Hash basic fields
        hash_combine(seed, turn_number);
        hash_combine(seed, active_player_id);
        hash_combine(seed, static_cast<int>(current_phase));

        // Hash players (all zones)
        for (const auto& player : players) {
            hash_combine(seed, player.id);

            // Helper for hashing a zone
            auto hash_zone = [&](const std::vector<CardInstance>& zone) {
                hash_combine(seed, zone.size());
                for (const auto& card : zone) {
                    hash_combine(seed, card.card_id);
                    hash_combine(seed, card.instance_id); // Including instance_id for strict state equality
                    hash_combine(seed, card.is_tapped);
                    hash_combine(seed, card.summoning_sickness);
                    hash_combine(seed, card.is_face_down);
                    hash_combine(seed, card.power_mod);
                    hash_combine(seed, card.cost_payment_meta);
                }
            };

            hash_zone(player.hand);
            hash_zone(player.mana_zone);
            hash_zone(player.battle_zone);
            hash_zone(player.graveyard);
            hash_zone(player.shield_zone);
            hash_zone(player.deck);
            hash_zone(player.hyper_spatial_zone);
            hash_zone(player.gr_deck);
        }

        // Hash pending effects
        hash_combine(seed, pending_effects.size());
        for (const auto& effect : pending_effects) {
            hash_combine(seed, static_cast<int>(effect.type));
            hash_combine(seed, effect.source_instance_id);
            hash_combine(seed, effect.controller);
            hash_combine(seed, effect.num_targets_needed);
            for (int tid : effect.target_instance_ids) {
                hash_combine(seed, tid);
            }
        }

        // Hash current attack
        hash_combine(seed, current_attack.source_instance_id);
        hash_combine(seed, current_attack.target_instance_id);
        hash_combine(seed, current_attack.target_player);
        hash_combine(seed, current_attack.is_blocked);
        hash_combine(seed, current_attack.blocker_instance_id);

        return static_cast<uint64_t>(seed);
    }
}
