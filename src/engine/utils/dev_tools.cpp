#include "dev_tools.hpp"
#include <algorithm>
#include <random>

namespace dm::engine {

    using namespace dm::core;

    int DevTools::move_cards(GameState& state, int player_id, Zone source, Zone target, int count, int card_id_filter) {
        if (player_id < 0 || player_id >= 2) return 0;
        if (count <= 0) return 0;
        if (source == target) return 0;

        Player& player = state.players[player_id];
        std::vector<CardInstance>* src_vec = nullptr;
        std::vector<CardInstance>* dst_vec = nullptr;

        auto get_zone_vec = [&](Zone z) -> std::vector<CardInstance>* {
            switch (z) {
                case Zone::DECK: return &player.deck;
                case Zone::HAND: return &player.hand;
                case Zone::MANA: return &player.mana_zone;
                case Zone::BATTLE: return &player.battle_zone;
                case Zone::GRAVEYARD: return &player.graveyard;
                case Zone::SHIELD: return &player.shield_zone;
                default: return nullptr;
            }
        };

        src_vec = get_zone_vec(source);
        dst_vec = get_zone_vec(target);

        if (!src_vec || !dst_vec) return 0;

        int moved_count = 0;
        // Iterate backwards to safely remove
        for (int i = src_vec->size() - 1; i >= 0 && moved_count < count; --i) {
            const auto& card = (*src_vec)[i];
            if (card_id_filter == -1 || card.card_id == card_id_filter) {
                // Move card
                CardInstance c = card;
                
                // Reset state when moving zones (usually)
                // But maybe we want to keep some state?
                // For dev tools, let's reset tap state unless moving to battle/mana?
                // Let's just copy and let the user handle state if needed, or reset to default.
                c.is_tapped = false;
                c.summoning_sickness = true; // Default for new zone usually

                dst_vec->push_back(c);
                src_vec->erase(src_vec->begin() + i);
                moved_count++;
            }
        }

        return moved_count;
    }

    void DevTools::trigger_loop_detection(GameState& state) {
        uint64_t current_hash = state.calculate_hash();
        // Push 2 times so the next update_loop_check finds 2 + 1(current) = 3
        state.hash_history.push_back(current_hash);
        state.hash_history.push_back(current_hash);
        // Explicitly trigger check to update winner status immediately
        state.update_loop_check();
    }

}
