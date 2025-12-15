#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/card_registry.hpp" // Added include for registry
#include <algorithm>

namespace dm::engine {

    // Move helper function to be a standalone static or member helper since GenericCardSystem::find_instance is not public/static
    static dm::core::CardInstance* find_instance_local(dm::core::GameState& game_state, int instance_id) {
        using namespace dm::core;
        for (auto& p : game_state.players) {
            for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.effect_buffer) if (c.instance_id == instance_id) return &c;
        }
        return nullptr;
    }

    class MoveCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            // Delegate if it requires explicit target selection
            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 GenericCardSystem::delegate_selection(ctx);
                 return;
            }

            // Implicit targets logic (similar to DestroyHandler)
            // Iterate all potential targets matching filter and move them.

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            // Determine zones to check
            std::vector<std::pair<PlayerID, Zone>> zones_to_check;

            // If filter.zones is specified, use it
            if (!ctx.action.filter.zones.empty()) {
                for (const auto& z_str : ctx.action.filter.zones) {
                    Zone z = Zone::BATTLE; // Default
                    if (z_str == "BATTLE_ZONE") z = Zone::BATTLE;
                    else if (z_str == "HAND") z = Zone::HAND;
                    else if (z_str == "MANA_ZONE") z = Zone::MANA;
                    else if (z_str == "SHIELD_ZONE") z = Zone::SHIELD;
                    else if (z_str == "GRAVEYARD") z = Zone::GRAVEYARD;
                    else if (z_str == "DECK") z = Zone::DECK; // Less common for implicit move but possible

                    std::vector<PlayerID> pids;
                    if (ctx.action.scope == TargetScope::ALL_PLAYERS) {
                         pids = {0, 1};
                    } else if (ctx.action.scope == TargetScope::PLAYER_OPPONENT) {
                         pids = { (PlayerID)(1 - controller_id) };
                    } else {
                         // Default to self for NONE/SELF
                         pids = { controller_id };
                    }

                    // Filter owner check overrides scope
                    if (ctx.action.filter.owner == "OPPONENT") {
                         pids = { (PlayerID)(1 - controller_id) };
                    } else if (ctx.action.filter.owner == "SELF") {
                         pids = { controller_id };
                    }

                    for (PlayerID pid : pids) {
                        zones_to_check.push_back({pid, z});
                    }
                }
            } else {
                // Default zones if not specified
                zones_to_check.push_back({controller_id, Zone::BATTLE});
                if (ctx.action.scope == TargetScope::ALL_PLAYERS) {
                    zones_to_check.push_back({(PlayerID)(1 - controller_id), Zone::BATTLE});
                }
            }

            std::vector<int> targets_to_move;

            for (const auto& [pid, zone] : zones_to_check) {
                Player& p = ctx.game_state.players[pid];
                const std::vector<CardInstance>* card_list = nullptr;

                // Map Zone enum to vector
                if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                else if (zone == Zone::HAND) card_list = &p.hand;
                else if (zone == Zone::MANA) card_list = &p.mana_zone;
                else if (zone == Zone::SHIELD) card_list = &p.shield_zone;
                else if (zone == Zone::GRAVEYARD) card_list = &p.graveyard;
                // Deck check? Iterating deck is expensive and rare for implicit moves.

                if (!card_list) continue;

                for (const auto& card : *card_list) {
                    const CardDefinition* def_ptr = nullptr;
                    if (ctx.card_db.count(card.card_id)) {
                        def_ptr = &ctx.card_db.at(card.card_id);
                    } else {
                        const auto& registry = CardRegistry::get_all_definitions();
                        if (registry.count(card.card_id)) {
                            def_ptr = &registry.at(card.card_id);
                        }
                    }
                    if (!def_ptr) continue;
                    const auto& def = *def_ptr;

                    if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                         // Check protections
                         if (pid != controller_id) {
                              if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                         }
                         targets_to_move.push_back(card.instance_id);
                    }
                }
            }

            // Apply movement
            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD"; // Default

            // Migrate to TransitionCommand
            Zone dest_zone = Zone::GRAVEYARD;
            if (dest == "HAND") dest_zone = Zone::HAND;
            else if (dest == "MANA_ZONE") dest_zone = Zone::MANA;
            else if (dest == "SHIELD_ZONE") dest_zone = Zone::SHIELD;
            else if (dest == "BATTLE_ZONE") dest_zone = Zone::BATTLE;
            else if (dest == "DECK" || dest == "DECK_TOP") dest_zone = Zone::DECK;
            else if (dest == "DECK_BOTTOM") dest_zone = Zone::DECK;

            int dest_idx = -1;
            if (dest == "DECK_BOTTOM") dest_idx = 0;

            for (int tid : targets_to_move) {
                auto* card = find_instance_local(ctx.game_state, tid);
                if (!card) continue;

                PlayerID owner = GenericCardSystem::get_controller(ctx.game_state, tid);
                Zone src_zone = Zone::GRAVEYARD; // Dummy default
                bool found = false;

                auto& p = ctx.game_state.players[owner];
                if (std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const auto& c){ return c.instance_id == tid; }) != p.battle_zone.end()) { src_zone = Zone::BATTLE; found = true; }
                else if (std::find_if(p.hand.begin(), p.hand.end(), [&](const auto& c){ return c.instance_id == tid; }) != p.hand.end()) { src_zone = Zone::HAND; found = true; }
                else if (std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const auto& c){ return c.instance_id == tid; }) != p.mana_zone.end()) { src_zone = Zone::MANA; found = true; }
                else if (std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const auto& c){ return c.instance_id == tid; }) != p.shield_zone.end()) { src_zone = Zone::SHIELD; found = true; }
                else if (std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const auto& c){ return c.instance_id == tid; }) != p.graveyard.end()) { src_zone = Zone::GRAVEYARD; found = true; }

                if (found) {
                     TransitionCommand cmd(tid, src_zone, dest_zone, owner, dest_idx);
                     cmd.execute(ctx.game_state);

                     if (src_zone == Zone::BATTLE) {
                         GenericCardSystem::check_mega_last_burst(ctx.game_state, *card, ctx.card_db);
                     }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            if (!ctx.targets) return;

            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD"; // Default

            Zone dest_zone = Zone::GRAVEYARD;
            if (dest == "HAND") dest_zone = Zone::HAND;
            else if (dest == "MANA_ZONE") dest_zone = Zone::MANA;
            else if (dest == "SHIELD_ZONE") dest_zone = Zone::SHIELD;
            else if (dest == "BATTLE_ZONE") dest_zone = Zone::BATTLE;
            else if (dest == "DECK" || dest == "DECK_TOP") dest_zone = Zone::DECK;
            else if (dest == "DECK_BOTTOM") dest_zone = Zone::DECK;

            int dest_idx = -1;
            if (dest == "DECK_BOTTOM") dest_idx = 0;

            for (int target_id : *ctx.targets) {
                auto* card = find_instance_local(ctx.game_state, target_id);
                if (!card) continue;

                PlayerID owner = GenericCardSystem::get_controller(ctx.game_state, target_id);
                Zone src_zone = Zone::GRAVEYARD;
                bool found = false;

                auto& p = ctx.game_state.players[owner];
                if (std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.battle_zone.end()) { src_zone = Zone::BATTLE; found = true; }
                else if (std::find_if(p.hand.begin(), p.hand.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.hand.end()) { src_zone = Zone::HAND; found = true; }
                else if (std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.mana_zone.end()) { src_zone = Zone::MANA; found = true; }
                else if (std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.shield_zone.end()) { src_zone = Zone::SHIELD; found = true; }
                else if (std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.graveyard.end()) { src_zone = Zone::GRAVEYARD; found = true; }
                else if (std::find_if(p.effect_buffer.begin(), p.effect_buffer.end(), [&](const auto& c){ return c.instance_id == target_id; }) != p.effect_buffer.end()) {
                    continue;
                }

                if (found) {
                     TransitionCommand cmd(target_id, src_zone, dest_zone, owner, dest_idx);
                     cmd.execute(ctx.game_state);

                     if (src_zone == Zone::BATTLE) {
                         GenericCardSystem::check_mega_last_burst(ctx.game_state, *card, ctx.card_db);
                     }
                }
            }
        }
    };
}
