#pragma once
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
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

            // Implicit targets logic
            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
            std::vector<std::pair<PlayerID, Zone>> zones_to_check;

            if (!ctx.action.filter.zones.empty()) {
                for (const auto& z_str : ctx.action.filter.zones) {
                    Zone z = Zone::BATTLE;
                    if (z_str == "BATTLE_ZONE") z = Zone::BATTLE;
                    else if (z_str == "HAND") z = Zone::HAND;
                    else if (z_str == "MANA_ZONE") z = Zone::MANA;
                    else if (z_str == "SHIELD_ZONE") z = Zone::SHIELD;
                    else if (z_str == "GRAVEYARD") z = Zone::GRAVEYARD;
                    else if (z_str == "DECK") z = Zone::DECK;

                    std::vector<PlayerID> pids;
                    if (ctx.action.scope == TargetScope::ALL_PLAYERS) {
                         pids = {0, 1};
                    } else if (ctx.action.scope == TargetScope::PLAYER_OPPONENT) {
                         pids = { (PlayerID)(1 - controller_id) };
                    } else {
                         pids = { controller_id };
                    }
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
                zones_to_check.push_back({controller_id, Zone::BATTLE});
                if (ctx.action.scope == TargetScope::ALL_PLAYERS) {
                    zones_to_check.push_back({(PlayerID)(1 - controller_id), Zone::BATTLE});
                }
            }

            std::vector<int> targets_to_move;

            for (const auto& [pid, zone] : zones_to_check) {
                Player& p = ctx.game_state.players[pid];
                const std::vector<CardInstance>* card_list = nullptr;

                if (zone == Zone::BATTLE) card_list = &p.battle_zone;
                else if (zone == Zone::HAND) card_list = &p.hand;
                else if (zone == Zone::MANA) card_list = &p.mana_zone;
                else if (zone == Zone::SHIELD) card_list = &p.shield_zone;
                else if (zone == Zone::GRAVEYARD) card_list = &p.graveyard;

                if (!card_list) continue;

                for (const auto& card : *card_list) {
                    if (!ctx.card_db.count(card.card_id)) continue;
                    const auto& def = ctx.card_db.at(card.card_id);

                    if (TargetUtils::is_valid_target(card, def, ctx.action.filter, ctx.game_state, controller_id, pid)) {
                         if (pid != controller_id) {
                              if (TargetUtils::is_protected_by_just_diver(card, def, ctx.game_state, controller_id)) continue;
                         }
                         targets_to_move.push_back(card.instance_id);
                    }
                }
            }

            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD";

            for (int tid : targets_to_move) {
                move_card_to_dest(ctx.game_state, tid, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets) return;

            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD";

            for (int target_id : *ctx.targets) {
                 move_card_to_dest(ctx.game_state, target_id, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

    private:
        void move_card_to_dest(dm::core::GameState& game_state, int instance_id, const std::string& dest, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            CardInstance card_copy;
            Zone from_zone = Zone::GRAVEYARD;
            PlayerID owner_id = 0;
            bool found = false;
            bool from_battle_zone = false;

             // Search all zones
            for (auto& p : game_state.players) {
                auto b_it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (b_it != p.battle_zone.end()) {
                    card_copy = *b_it;
                    from_zone = Zone::BATTLE;
                    owner_id = p.id;
                    found = true;
                    from_battle_zone = true;
                    break;
                }
                auto h_it = std::find_if(p.hand.begin(), p.hand.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (h_it != p.hand.end()) {
                    card_copy = *h_it;
                    from_zone = Zone::HAND;
                    owner_id = p.id;
                    found = true;
                    break;
                }
                auto m_it = std::find_if(p.mana_zone.begin(), p.mana_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (m_it != p.mana_zone.end()) {
                    card_copy = *m_it;
                    from_zone = Zone::MANA;
                    owner_id = p.id;
                    found = true;
                    break;
                }
                auto s_it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (s_it != p.shield_zone.end()) {
                    card_copy = *s_it;
                    from_zone = Zone::SHIELD;
                    owner_id = p.id;
                    found = true;
                    break;
                }
                auto g_it = std::find_if(p.graveyard.begin(), p.graveyard.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (g_it != p.graveyard.end()) {
                    card_copy = *g_it;
                    from_zone = Zone::GRAVEYARD;
                    owner_id = p.id;
                    found = true;
                    break;
                }
                auto d_it = std::find_if(p.deck.begin(), p.deck.end(), [&](const CardInstance& c){ return c.instance_id == instance_id; });
                if (d_it != p.deck.end()) {
                    card_copy = *d_it;
                    from_zone = Zone::DECK;
                    owner_id = p.id;
                    found = true;
                    break;
                }
            }

            if (!found) return;

            // Pre-move logic: Battle Zone leave triggers
            if (from_battle_zone) {
                ZoneUtils::on_leave_battle_zone(game_state, card_copy);
            }

            Zone to_zone = Zone::GRAVEYARD;
            int dest_idx = -1;

            if (dest == "SHIELD_ZONE") {
                 to_zone = Zone::SHIELD;
            } else if (dest == "HAND") {
                 to_zone = Zone::HAND;
            } else if (dest == "MANA_ZONE") {
                 to_zone = Zone::MANA;
            } else if (dest == "GRAVEYARD") {
                 to_zone = Zone::GRAVEYARD;
            } else if (dest == "DECK_BOTTOM") {
                 to_zone = Zone::DECK;
                 dest_idx = 0;
            } else if (dest == "DECK_TOP") {
                 to_zone = Zone::DECK;
                 dest_idx = -1;
            } else if (dest == "BATTLE_ZONE") {
                 to_zone = Zone::BATTLE;
            }

            TransitionCommand cmd(instance_id, from_zone, to_zone, owner_id, dest_idx);
            cmd.execute(game_state);

            // Post-move triggers
            if (dest == "SHIELD_ZONE") {
                 GenericCardSystem::resolve_trigger(game_state, TriggerType::ON_SHIELD_ADD, instance_id, card_db);
            }

            // Mega Last Burst check
            if (from_battle_zone) {
                 // Pass the copy we captured before the move.
                 GenericCardSystem::check_mega_last_burst(game_state, card_copy, card_db);
            }
        }
    };
}
