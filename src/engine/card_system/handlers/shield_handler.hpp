#pragma once
#include "../effect_system.hpp"
#include "core/game_state.hpp"
#include "../generic_card_system.hpp"
#include <algorithm>
#include <vector>
#include <set>

namespace dm::engine {

    class ShieldHandler : public IActionHandler {
    public:
        void resolve(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) override {
            using namespace dm::core;

            PlayerID controller_id = GenericCardSystem::get_controller(game_state, source_instance_id);
            Player& controller = game_state.players[controller_id];

            if (action.type == EffectActionType::ADD_SHIELD) {
                std::vector<CardInstance>* source = &controller.deck;
                if (action.source_zone == "HAND") source = &controller.hand;
                else if (action.source_zone == "GRAVEYARD") source = &controller.graveyard;

                if (!source->empty()) {
                    CardInstance c = source->back();
                    source->pop_back();
                    c.is_face_down = true;
                    controller.shield_zone.push_back(c);
                }
            } else if (action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {
                if (action.scope != TargetScope::TARGET_SELECT && action.target_choice != "SELECT") {
                     if (!controller.shield_zone.empty()) {
                        CardInstance c = controller.shield_zone.back();
                        controller.shield_zone.pop_back();
                        controller.graveyard.push_back(c);
                    }
                } else {
                     EffectDef ed;
                     ed.trigger = TriggerType::NONE;
                     ed.condition = ConditionDef{"NONE", 0, ""};
                     ed.actions = { action };
                     GenericCardSystem::select_targets(game_state, action, source_instance_id, ed, execution_context);
                }
            }
        }

        void resolve_with_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, const std::vector<int>& targets, int source_id, std::map<std::string, int>& context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) override {
            using namespace dm::core;
             if (action.type == EffectActionType::SEND_SHIELD_TO_GRAVE) {

                 std::set<int> target_ids(targets.begin(), targets.end());
                 std::vector<int> final_targets;

                 if (action.inverse_target) {
                     // Collect all shields from targeted players
                     // We need to know WHOSE shields were targeted.
                     // Filter owner determines it.
                     std::vector<int> players_to_check;
                     PlayerID decision_maker = game_state.active_player_id; // Default assumption for context
                     // Actually GenericCardSystem::select_targets uses owner filter.
                     // Here we re-evaluate filter to find universe.

                     if (action.filter.owner.has_value()) {
                        std::string req = action.filter.owner.value();
                        // "SELF" relative to whom? Usually source controller.
                        PlayerID controller = GenericCardSystem::get_controller(game_state, source_id);
                        if (req == "SELF") players_to_check.push_back(controller);
                        else if (req == "OPPONENT") players_to_check.push_back(1 - controller);
                        else if (req == "BOTH") { players_to_check.push_back(controller); players_to_check.push_back(1 - controller); }
                     } else {
                         // Default to opponent for "Send shield to grave"? Or self?
                         // Usually shield removal targets opponent.
                         // But if manual select, the targets tell us.
                         // Let's iterate all players and check if their shields are in target set?
                         // No, we need to know the scope to invert.
                         // If I targeted Opponent's shields, I want to invert Opponent's shields.
                         // Not mine.

                         // Heuristic: If targets are empty, we can't infer from targets.
                         // But if targets provided, we can see who owns them?
                         // But if user selected 0 targets (optional), we still need to know scope to destroy ALL.
                         // So we MUST rely on filter.

                         // Fallback: If filter owner not set, assume Opponent?
                         // Or iterate both.
                         players_to_check.push_back(0);
                         players_to_check.push_back(1);
                     }

                     for (int pid : players_to_check) {
                         const auto& p = game_state.players[pid];
                         for (const auto& s : p.shield_zone) {
                             if (target_ids.find(s.instance_id) == target_ids.end()) {
                                 // Not selected -> Destroy
                                 final_targets.push_back(s.instance_id);
                             }
                         }
                     }
                 } else {
                     final_targets = targets;
                 }

                 for (int tid : final_targets) {
                    for (auto &p : game_state.players) {
                         auto it = std::find_if(p.shield_zone.begin(), p.shield_zone.end(),
                            [tid](const CardInstance& c){ return c.instance_id == tid; });
                         if (it != p.shield_zone.end()) {
                             p.graveyard.push_back(*it);
                             p.shield_zone.erase(it);
                             break;
                         }
                    }
                }
             }
        }
    };
}
