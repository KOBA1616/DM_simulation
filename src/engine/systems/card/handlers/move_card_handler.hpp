#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/selection_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>

namespace dm::engine {

    class MoveCardHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            std::vector<int> targets;

            if (ctx.targets && !ctx.targets->empty()) {
                targets = *ctx.targets;
            } else if (ctx.action.target_choice != "SELECT") {
                // Determine implicit targets
                // Copy selection logic from resolve()
                PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
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
                        // filter owner overrides scope
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
                             targets.push_back(card.instance_id);
                        }
                    }
                }
            }

            // Handle implicit "Draw" logic: if no targets selected and source is DECK, pick top N
            if (targets.empty() && ctx.action.source_zone == "DECK") {
                int count = ctx.action.value1;
                if (count <= 0) count = 1;
                // Cap count at deck size
                PlayerID controller_id = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                // Check if scope specifies implicit target player different from controller?
                // Usually ActionDef.target_player or scope determines it.
                // For simplified logic, assume implicit Draw is for controller or derived scope.

                // If filter specified zones, we might have checked them above.
                // But for pure "Draw 2" action, source_zone="DECK" and filter might be empty.
                // We need to fetch deck cards.

                std::vector<PlayerID> pids;
                 if (ctx.action.scope == TargetScope::ALL_PLAYERS) {
                        pids = {0, 1};
                } else if (ctx.action.scope == TargetScope::PLAYER_OPPONENT) {
                        pids = { (PlayerID)(1 - controller_id) };
                } else {
                        // Default to self for Draw
                        pids = { controller_id };
                }
                 if (ctx.action.filter.owner == "OPPONENT") {
                        pids = { (PlayerID)(1 - controller_id) };
                } else if (ctx.action.filter.owner == "SELF") {
                        pids = { controller_id };
                }

                for (PlayerID pid : pids) {
                     Player& p = ctx.game_state.players[pid];
                     int available = p.deck.size();
                     int to_take = std::min(count, available);
                     // Take from top (end of vector)
                     for (int i = 0; i < to_take; ++i) {
                          // index: size - 1 - i
                          if (p.deck.empty()) break;
                          // Better to just push back the ID. We assume deck structure is valid.
                          // However, we need Instance ID.
                          // Deck stores CardInstance.
                          int idx = p.deck.size() - 1 - i;
                          if (idx >= 0) targets.push_back(p.deck[idx].instance_id);
                     }
                }
            }

            if (targets.empty()) return;

            std::string dest = ctx.action.destination_zone;
            if (dest.empty()) dest = "GRAVEYARD";

            // Map destination to internal strings expected by handle_move
            // "GRAVEYARD" -> "GRAVEYARD"
            // "HAND" -> "HAND"
            // "MANA_ZONE" -> "MANA"
            // "SHIELD_ZONE" -> "SHIELD"
            // "DECK_BOTTOM" -> "DECK" (with to_bottom=true)
            // "DECK_TOP" -> "DECK"
            // "BATTLE_ZONE" -> "BATTLE"

            std::string to_val = "GRAVEYARD";
            bool to_bottom = false;

            if (dest == "SHIELD_ZONE") to_val = "SHIELD";
            else if (dest == "HAND") to_val = "HAND";
            else if (dest == "MANA_ZONE") to_val = "MANA";
            else if (dest == "GRAVEYARD") to_val = "GRAVEYARD";
            else if (dest == "DECK_BOTTOM") { to_val = "DECK"; to_bottom = true; }
            else if (dest == "DECK_TOP") { to_val = "DECK"; }
            else if (dest == "BATTLE_ZONE") { to_val = "BATTLE"; }
            else if (dest == "STACK") { to_val = "STACK"; }

            for (int t : targets) {
                 nlohmann::json move_args;
                 move_args["target"] = t;
                 move_args["to"] = to_val;
                 if (to_bottom) move_args["to_bottom"] = true;
                 ctx.instruction_buffer->emplace_back(InstructionOp::MOVE, move_args);

                 // Handle specific post-move triggers via instruction?
                 // TransitionCommand handles basic triggers.
                 // Mega Last Burst logic is complex (depends on leaving battle zone).
                 // TransitionCommand handles "on_leave_battle_zone" if it knows previous zone.
                 // TransitionCommand implementation in existing code checks pre-move state for on_leave_battle.
                 // So we should be covered.
                 // However, "ON_SHIELD_ADD" trigger needs to be checked.
                 // TransitionCommand usually emits "ZONE_ENTER" events.
                 // TriggerManager should listen to these events.
                 // If not fully automatic, we might need explicit trigger instructions.
                 // For now, assuming Pipeline/Command robustness.
            }
        }

        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

            // Delegate if it requires explicit target selection
            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 SelectionSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile_action(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }
    };
}
