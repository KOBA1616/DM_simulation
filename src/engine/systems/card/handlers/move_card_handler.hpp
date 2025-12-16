#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/utils/zone_utils.hpp"
#include <algorithm>
#include <memory>

namespace dm::engine {

    class MoveCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.trigger = dm::core::TriggerType::NONE;
                 ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }

            std::string dest = ctx.action.str_val;
            int dest_idx = ctx.action.value2;

            PlayerID controller_id = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            std::vector<int> targets;
            if (ctx.targets) {
                targets = *ctx.targets;
            }

            for (int tid : targets) {
                move_card_to_dest(ctx.game_state, tid, dest, ctx.source_instance_id, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
             if (!ctx.targets) return;
             std::string dest = ctx.action.str_val;
             if (dest.empty()) return;

             for (int tid : *ctx.targets) {
                 move_card_to_dest(ctx.game_state, tid, dest, ctx.source_instance_id, ctx.card_db);
             }
        }

    private:
        void move_card_to_dest(dm::core::GameState& game_state, int instance_id, const std::string& dest, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            Zone from_zone = Zone::GRAVEYARD;
            PlayerID owner_id = 0;
            bool found = false;

            for (int pid = 0; pid < 2; ++pid) {
                const Player& p = game_state.players[pid];
                for(const auto& c : p.hand) if(c.instance_id == instance_id) { from_zone=Zone::HAND; owner_id=pid; found=true; break; }
                if(found) break;
                for(const auto& c : p.battle_zone) if(c.instance_id == instance_id) { from_zone=Zone::BATTLE; owner_id=pid; found=true; break; }
                if(found) break;
                for(const auto& c : p.mana_zone) if(c.instance_id == instance_id) { from_zone=Zone::MANA; owner_id=pid; found=true; break; }
                if(found) break;
                for(const auto& c : p.graveyard) if(c.instance_id == instance_id) { from_zone=Zone::GRAVEYARD; owner_id=pid; found=true; break; }
                if(found) break;
                for(const auto& c : p.shield_zone) if(c.instance_id == instance_id) { from_zone=Zone::SHIELD; owner_id=pid; found=true; break; }
                if(found) break;
                for(const auto& c : game_state.stack_zone) if(c.instance_id == instance_id) { from_zone=Zone::STACK; owner_id=pid; found=true; break; }
                if(found) break;
            }

            if (!found) {
                for(const auto& c : game_state.stack_zone) {
                    if(c.instance_id == instance_id) {
                         from_zone = Zone::STACK;
                         owner_id = c.owner;
                         found = true;
                         break;
                    }
                }
            }

            if (!found) return;

            Zone to_zone = Zone::GRAVEYARD;
            if (dest == "HAND") to_zone = Zone::HAND;
            else if (dest == "MANA") to_zone = Zone::MANA;
            else if (dest == "GRAVEYARD") to_zone = Zone::GRAVEYARD;
            else if (dest == "SHIELD") to_zone = Zone::SHIELD;
            else if (dest == "DECK") to_zone = Zone::DECK;
            else if (dest == "BATTLE_ZONE") to_zone = Zone::BATTLE;

            if (from_zone == Zone::BATTLE && to_zone != Zone::BATTLE) {
                 CardInstance* ptr = game_state.get_card_instance(instance_id);
                 if (ptr) ZoneUtils::on_leave_battle_zone(game_state, *ptr);
            }

            // Use shared_ptr and execute_command for Undo
            auto cmd = std::make_shared<TransitionCommand>(instance_id, static_cast<int>(from_zone), static_cast<int>(to_zone), owner_id, -1);
            game_state.execute_command(cmd);
        }
    };
}
