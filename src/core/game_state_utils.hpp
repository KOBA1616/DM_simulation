#pragma once
#include "game_state.hpp"

namespace dm::core {
    inline Zone get_zone_type(const GameState& state, int instance_id) {
        if(instance_id < 0 || instance_id >= (int)state.card_owner_map.size()) return Zone::GRAVEYARD; // Default/Fallback
        PlayerID pid = state.card_owner_map[instance_id];
        if(pid >= state.players.size()) return Zone::GRAVEYARD;

        const auto& p = state.players[pid];
        auto contains = [&](const std::vector<CardInstance>& v) {
            for(const auto& c : v) if(c.instance_id == instance_id) return true;
            return false;
        };

        if(contains(p.hand)) return Zone::HAND;
        if(contains(p.mana_zone)) return Zone::MANA;
        if(contains(p.graveyard)) return Zone::GRAVEYARD;
        if(contains(p.battle_zone)) return Zone::BATTLE;
        if(contains(p.shield_zone)) return Zone::SHIELD;
        if(contains(p.deck)) return Zone::DECK;
        if(contains(p.effect_buffer)) return Zone::BUFFER;
        if(contains(p.stack)) return Zone::STACK;
        if(contains(p.hyper_spatial_zone)) return Zone::HYPER_SPATIAL;
        if(contains(p.gr_deck)) return Zone::GR_DECK;

        return Zone::GRAVEYARD; // Fallback
    }
}
