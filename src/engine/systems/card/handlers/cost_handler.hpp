#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/game_command/commands.hpp"

namespace dm::engine {

    class CostHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (ctx.action.type == EffectActionType::COST_REFERENCE) {
                if (ctx.action.str_val == "FINISH_HYPER_ENERGY") {
                     if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                         EffectDef ed;
                         ed.trigger = TriggerType::NONE;
                         ed.condition = ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                         ed.actions = { ctx.action };
                         EffectSystem::instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                     }
                }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             using namespace dm::core;
             using namespace dm::engine::systems;

             if (!ctx.targets) return;

             if (ctx.action.type == EffectActionType::COST_REFERENCE && ctx.action.str_val == "FINISH_HYPER_ENERGY") {
                 for (int tid : *ctx.targets) {
                     for (auto &p : ctx.game_state.players) {
                          auto it = std::find_if(p.battle_zone.begin(), p.battle_zone.end(),
                             [tid](const CardInstance& c){ return c.instance_id == tid; });
                          if (it != p.battle_zone.end()) {
                              it->is_tapped = true;
                              // Should likely dispatch event or use MutateCommand if strict
                          }
                     }
                 }
                 // int taps = ctx.action.value1; // unused logic?
                 // int reduction = taps * 2; // handled by auto-pay logic usually or here.
                 // In Hyper Energy, we just tapped creatures.
                 // Now resolve play.

                 PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

                 // Use GameLogicSystem
                 Action resolve_act;
                 resolve_act.type = ActionType::RESOLVE_PLAY;
                 resolve_act.source_instance_id = ctx.source_instance_id;
                 resolve_act.spawn_source = SpawnSource::HAND_SUMMON; // Hyper Energy is from hand

                 GameLogicSystem::resolve_action_oneshot(ctx.game_state, resolve_act, ctx.card_db);
             }
        }
    };
}
