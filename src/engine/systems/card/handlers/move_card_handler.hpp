#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <algorithm>
#include <vector>

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
        void move_card_to_dest(dm::core::GameState& game_state, int instance_id, const std::string& dest, int /*source_instance_id*/, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            // 1. Identify Source
            Zone from_zone = Zone::GRAVEYARD; // Default, will update
            PlayerID owner_id = 0;
            bool found = false;
            bool from_battle_zone = false;

            // Search to find 'from_zone' and 'owner_id'
            for (auto& p : game_state.players) {
                if (find_in_zone(p.battle_zone, instance_id)) { from_zone = Zone::BATTLE; owner_id = p.id; found = true; from_battle_zone = true; break; }
                if (find_in_zone(p.hand, instance_id)) { from_zone = Zone::HAND; owner_id = p.id; found = true; break; }
                if (find_in_zone(p.mana_zone, instance_id)) { from_zone = Zone::MANA; owner_id = p.id; found = true; break; }
                if (find_in_zone(p.shield_zone, instance_id)) { from_zone = Zone::SHIELD; owner_id = p.id; found = true; break; }
                if (find_in_zone(p.graveyard, instance_id)) { from_zone = Zone::GRAVEYARD; owner_id = p.id; found = true; break; }
                if (find_in_zone(p.deck, instance_id)) { from_zone = Zone::DECK; owner_id = p.id; found = true; break; }
            }

            if (!found) {
                // Buffer check
                for (auto& p : game_state.players) {
                     if (find_in_zone(p.effect_buffer, instance_id)) {
                         // Fallback for buffer (not supported by TransitionCommand)
                         auto it = std::find_if(p.effect_buffer.begin(), p.effect_buffer.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                         if (it != p.effect_buffer.end()) {
                             CardInstance card = *it;
                             p.effect_buffer.erase(it);

                             // Replicate legacy destination logic manually
                             Player& owner = game_state.players[p.id];

                             if (dest == "SHIELD_ZONE") {
                                 card.is_tapped = false;
                                 owner.shield_zone.push_back(card);
                                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_SHIELD_ADD, card.instance_id, card_db);
                             } else if (dest == "HAND") {
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
                         }
                         return;
                     }
                }
                return; // Not found anywhere
            }

            // 2. Identify Destination Zone and Index
            Zone to_zone = Zone::GRAVEYARD;
            int dest_index = -1; // Default append

            if (dest == "SHIELD_ZONE") to_zone = Zone::SHIELD;
            else if (dest == "HAND") to_zone = Zone::HAND;
            else if (dest == "MANA_ZONE") to_zone = Zone::MANA;
            else if (dest == "GRAVEYARD") to_zone = Zone::GRAVEYARD;
            else if (dest == "DECK_BOTTOM") { to_zone = Zone::DECK; dest_index = 0; }
            else if (dest == "DECK_TOP") { to_zone = Zone::DECK; dest_index = -1; }
            else if (dest == "BATTLE_ZONE") to_zone = Zone::BATTLE;

            // 3. Pre-move Side Effects
            if (from_battle_zone) {
                 CardInstance* c = game_state.get_card_instance(instance_id);
                 if (c) ZoneUtils::on_leave_battle_zone(game_state, *c);
            }

            // 4. Execute Command
            TransitionCommand cmd(instance_id, from_zone, to_zone, owner_id, dest_index);
            cmd.execute(game_state);

            // 5. Post-move Side Effects and Property Resets
            // Retrieve card from new location
            CardInstance* moved_card = game_state.get_card_instance(instance_id);
            if (!moved_card) return; // Should not happen

            // Reset properties
            moved_card->is_tapped = false; // Default reset for all moves

            if (dest == "HAND") {
                moved_card->summoning_sickness = true;
            } else if (dest == "BATTLE_ZONE") {
                moved_card->summoning_sickness = true;
                moved_card->turn_played = game_state.turn_number;
            } else if (dest == "SHIELD_ZONE") {
                 // Trigger ON_SHIELD_ADD
                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_SHIELD_ADD, instance_id, card_db);
            }

            if (from_battle_zone) {
                 GenericCardSystem::check_mega_last_burst(game_state, *moved_card, card_db);
            }
        }

        bool find_in_zone(const std::vector<dm::core::CardInstance>& zone, int instance_id) {
            for (const auto& c : zone) {
                if (c.instance_id == instance_id) return true;
            }
            return false;
        }
    };
}
