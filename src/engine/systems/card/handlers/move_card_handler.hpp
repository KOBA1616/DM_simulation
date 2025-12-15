#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"
#include <algorithm>

namespace dm::engine {

    class MoveCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Delegate if it requires explicit target selection
            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
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
                    if (!ctx.card_db.count(card.card_id)) continue;
                    const auto& def = ctx.card_db.at(card.card_id);

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

            for (int tid : targets_to_move) {
                move_card_to_dest(ctx.game_state, tid, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets) return;

            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD"; // Default

            for (int target_id : *ctx.targets) {
                 move_card_to_dest(ctx.game_state, target_id, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

    private:
        void move_card_to_dest(dm::core::GameState& game_state, int instance_id, const std::string& dest, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;

            CardInstance card;
            bool found = false;
            bool from_battle_zone = false;
            PlayerID owner_id = 0;

            // Search all zones
            for (auto& p : game_state.players) {
                // Battle Zone
                auto b_it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (b_it != p.battle_zone.end()) {
                    card = *b_it;
                    ZoneUtils::on_leave_battle_zone(game_state, *b_it);
                    p.battle_zone.erase(b_it);
                    found = true;
                    from_battle_zone = true;
                    owner_id = p.id;
                    break;
                }

                // Hand
                auto h_it = std::find_if(p.hand.begin(), p.hand.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (h_it != p.hand.end()) {
                    card = *h_it;
                    p.hand.erase(h_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Mana
                auto m_it = std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (m_it != p.mana_zone.end()) {
                    card = *m_it;
                    p.mana_zone.erase(m_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Shield
                auto s_it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (s_it != p.shield_zone.end()) {
                    card = *s_it;
                    p.shield_zone.erase(s_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Graveyard
                auto g_it = std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (g_it != p.graveyard.end()) {
                    card = *g_it;
                    p.graveyard.erase(g_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }

                // Deck
                 auto d_it = std::find_if(p.deck.begin(), p.deck.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (d_it != p.deck.end()) {
                    card = *d_it;
                    p.deck.erase(d_it);
                    found = true;
                    owner_id = p.id;
                    break;
                }
            }

            if (!found) {
                // Check buffers
                for (auto& p : game_state.players) {
                    auto buf_it = std::find_if(p.effect_buffer.begin(), p.effect_buffer.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                    if (buf_it != p.effect_buffer.end()) {
                        card = *buf_it;
                        p.effect_buffer.erase(buf_it);
                        found = true;
                        owner_id = p.id;
                        break;
                    }
                }
            }

            if (!found) return;

            // Destination
            Player& owner = game_state.players[owner_id];

            // Special handling for SHIELD_ZONE (Shield Trigger / Shield Addition)
            if (dest == "SHIELD_ZONE") {
                 card.is_tapped = false;
                 owner.shield_zone.push_back(card);
                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_SHIELD_ADD, card.instance_id, card_db);
            }
            // General handling
            else if (dest == "HAND") {
                card.is_tapped = false;
                card.summoning_sickness = true;
                owner.hand.push_back(card);
            } else if (dest == "MANA_ZONE") {
                card.is_tapped = false;
                owner.mana_zone.push_back(card);
            } else if (dest == "GRAVEYARD") {
                card.is_tapped = false;
                owner.graveyard.push_back(card);
            } else if (dest == "DECK_BOTTOM") {
                card.is_tapped = false;
                owner.deck.insert(owner.deck.begin(), card);
            } else if (dest == "DECK_TOP") {
                card.is_tapped = false;
                owner.deck.push_back(card);
            } else if (dest == "BATTLE_ZONE") {
                card.is_tapped = false;
                card.summoning_sickness = true;
                card.turn_played = game_state.turn_number;
                owner.battle_zone.push_back(card);
            }

            if (from_battle_zone) {
                 GenericCardSystem::check_mega_last_burst(game_state, card, card_db);
            }
        }
    };
}
