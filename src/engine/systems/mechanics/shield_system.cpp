#include "shield_system.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/systems/effects/trigger_system.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/systems/breaker/breaker_system.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    void ShieldSystem::handle_break_shield(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                           const std::map<CardID, CardDefinition>& card_db) {
        std::vector<int> shield_ids;
        if (inst.args.find("shields") != inst.args.end()) {
            try {
                for (auto &v : inst.args["shields"]) shield_ids.push_back((int)v);
            } catch (...) { }
        } else {
            int single = exec.resolve_int(inst.args.value("shield", -1));
            if (single != -1) shield_ids.push_back(single);
        }

        if (shield_ids.empty()) return;

        std::vector<Instruction> generated;

        // 0. Check Before Break Triggers
        int source_id = exec.resolve_int(inst.args.value("source_id", -1));
        if (source_id != -1) {
             auto effects = TriggerSystem::instance().get_trigger_effects(state, TriggerType::BEFORE_BREAK_SHIELD, source_id, card_db);
             if (!effects.empty()) {
                 std::map<std::string, int> ctx;
                 for (const auto& eff : effects) {
                     dm::engine::effects::EffectSystem::instance().resolve_effect_with_context(state, eff, source_id, ctx, card_db);
                 }
             }
        }

        // 1. Buffer Shields
        std::vector<int> to_apply_list;
        for (int shield_id : shield_ids) {
            bool found = false;
            for (auto pid = 0u; pid < state.players.size(); ++pid) {
                auto &vec = state.players[pid].shield_zone;
                auto it = std::find_if(vec.begin(), vec.end(), [shield_id](const CardInstance& c){ return c.instance_id == shield_id; });
                if (it != vec.end()) {
                    CardInstance ci = *it;
                    vec.erase(it);
                    state.players[pid].effect_buffer.push_back(ci);
                    found = true;
                    break;
                }
            }
            if (!found) continue;

            to_apply_list.push_back(shield_id);

            const auto* card = state.get_card_instance(shield_id);
            if (card && card_db.count(card->card_id)) {
                const auto& def = card_db.at(card->card_id);
                if (def.keywords.shield_trigger) {
                    Instruction check(InstructionOp::GAME_ACTION);
                    check.args["type"] = "CHECK_S_TRIGGER";
                    check.args["card"] = shield_id;
                    generated.push_back(check);
                }
            }
        }

        // 2. Apply Move
        if (!to_apply_list.empty()) {
            nlohmann::json apply_args;
            apply_args["type"] = "APPLY_BUFFER_MOVE";
            apply_args["shields"] = to_apply_list;
            generated.emplace_back(InstructionOp::GAME_ACTION, apply_args);
        }

        if (!generated.empty()) {
             auto block = std::make_shared<std::vector<Instruction>>(generated);
             exec.call_stack.push_back({block, 0, LoopContext{}});
        }
    }

    void ShieldSystem::check_s_trigger(PipelineExecutor& exec, GameState& state, const Instruction& inst,
                                       const std::map<CardID, CardDefinition>& card_db) {
       int card_id = exec.resolve_int(inst.args.value("card", -1));

       std::string decision_key = "$strigger_" + std::to_string(card_id);
       ContextValue val = exec.get_context_var(decision_key);

       bool decided = false;
       if (std::holds_alternative<int>(val)) {
           decided = true;
       }

       if (!decided) {
           exec.execution_paused = true;
           exec.waiting_for_key = decision_key;

           std::vector<std::string> opts = {"No", "Shield Trigger", "Guard Strike", "Strike Back"};
           auto cmd = std::make_unique<QueryCommand>("SELECT_OPTION", std::vector<int>{}, std::map<std::string, int>{}, opts);
           state.execute_command(std::move(cmd));
           return;
       }

       if (std::holds_alternative<int>(val)) {
           int sel = std::get<int>(val);
           // 0=No, 1=Shield Trigger, 2=Guard Strike, 3=Strike Back
           if (sel == 1) {
               exec.set_context_var(decision_key, 1);
           } else if (sel == 2) {
               std::string k = "$guard_" + std::to_string(card_id);
               exec.set_context_var(k, 1);
           } else if (sel == 3) {
               std::string k = "$sback_" + std::to_string(card_id);
               exec.set_context_var(k, 1);
           } else {
               exec.set_context_var(decision_key, 0);
           }
       }

       ContextValue post = exec.get_context_var(decision_key);
       if (std::holds_alternative<int>(post) && std::get<int>(post) == 1) {
           (void)card_db;
           Instruction play_inst(InstructionOp::PLAY);
           play_inst.args["card"] = card_id;

           auto block = std::make_shared<std::vector<Instruction>>();
           block->push_back(play_inst);
           exec.call_stack.push_back({block, 0, LoopContext{}});
       }
    }

    int ShieldSystem::get_breaker_count(const GameState& state, const CardInstance& creature, const std::map<CardID, CardDefinition>& card_db) {
        if (!card_db.count(creature.card_id)) return 1;
        const auto& def = card_db.at(creature.card_id);
        return BreakerSystem::get_breaker_count(state, creature, def);
    }

}
