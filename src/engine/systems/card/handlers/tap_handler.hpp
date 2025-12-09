#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"

namespace dm::engine {

    class TapHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (ctx.action.scope == TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 EffectDef ed;
                 ed.trigger = TriggerType::NONE;
                 ed.condition = ConditionDef{"NONE", 0, ""};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            if (ctx.action.target_choice == "ALL_ENEMY") {
                 int controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 int enemy = 1 - controller_id;
                 for (auto& c : ctx.game_state.players[enemy].battle_zone) {
                     c.is_tapped = true;
                 }
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             if (!ctx.targets) return;

             auto find_inst = [&](int instance_id) -> dm::core::CardInstance* {
                for (auto& p : ctx.game_state.players) {
                    for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
                    for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
                }
                return nullptr;
            };

            for (int tid : *ctx.targets) {
                dm::core::CardInstance* inst = find_inst(tid);
                if (inst) inst->is_tapped = true;
            }
        }
    };
}
