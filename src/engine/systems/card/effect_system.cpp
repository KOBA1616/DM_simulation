#include "effect_system.hpp"
#include "card_registry.hpp"
#include "target_utils.hpp"
#include "condition_system.hpp"
#include "handlers/draw_handler.hpp"
#include "handlers/mana_handler.hpp"
#include "handlers/destroy_handler.hpp"
#include "handlers/return_to_hand_handler.hpp"
#include "handlers/tap_handler.hpp"
#include "handlers/untap_handler.hpp"
#include "handlers/count_handler.hpp"
#include "handlers/shield_handler.hpp"
#include "handlers/search_handler.hpp"
#include "handlers/buffer_handler.hpp"
#include "handlers/cost_handler.hpp"
#include "handlers/hierarchy_handler.hpp"
#include "handlers/reveal_handler.hpp"
#include "handlers/select_number_handler.hpp"
#include "handlers/friend_burst_handler.hpp"
#include "handlers/grant_keyword_handler.hpp"
#include "handlers/move_card_handler.hpp"
#include "handlers/cast_spell_handler.hpp"
#include "handlers/put_creature_handler.hpp"
#include "handlers/modifier_handler.hpp"
#include "handlers/select_option_handler.hpp"
#include "handlers/break_shield_handler.hpp"
#include "handlers/discard_handler.hpp"
#include "handlers/play_handler.hpp"
#include "handlers/modify_power_handler.hpp"
#include "handlers/resolve_battle_handler.hpp"
#include "engine/systems/command_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    void EffectSystem::initialize() {
        if (initialized) return;

        register_handler(EffectPrimitive::DRAW_CARD, std::make_unique<DrawHandler>());
        register_handler(EffectPrimitive::ADD_MANA, std::make_unique<ManaChargeHandler>());
        register_handler(EffectPrimitive::SEND_TO_MANA, std::make_unique<ManaChargeHandler>());
        register_handler(EffectPrimitive::DESTROY, std::make_unique<DestroyHandler>());
        register_handler(EffectPrimitive::RETURN_TO_HAND, std::make_unique<ReturnToHandHandler>());
        register_handler(EffectPrimitive::TAP, std::make_unique<TapHandler>());
        register_handler(EffectPrimitive::UNTAP, std::make_unique<UntapHandler>());
        register_handler(EffectPrimitive::COUNT_CARDS, std::make_unique<CountHandler>());
        register_handler(EffectPrimitive::GET_GAME_STAT, std::make_unique<CountHandler>());
        register_handler(EffectPrimitive::ADD_SHIELD, std::make_unique<ShieldHandler>());
        register_handler(EffectPrimitive::SEND_SHIELD_TO_GRAVE, std::make_unique<ShieldHandler>());
        register_handler(EffectPrimitive::SEARCH_DECK, std::make_unique<SearchHandler>());
        register_handler(EffectPrimitive::SEARCH_DECK_BOTTOM, std::make_unique<SearchHandler>());
        register_handler(EffectPrimitive::SEND_TO_DECK_BOTTOM, std::make_unique<SearchHandler>());
        register_handler(EffectPrimitive::SHUFFLE_DECK, std::make_unique<SearchHandler>());
        register_handler(EffectPrimitive::MEKRAID, std::make_unique<BufferHandler>());
        register_handler(EffectPrimitive::LOOK_TO_BUFFER, std::make_unique<BufferHandler>());
        register_handler(EffectPrimitive::MOVE_BUFFER_TO_ZONE, std::make_unique<BufferHandler>());
        register_handler(EffectPrimitive::PLAY_FROM_BUFFER, std::make_unique<BufferHandler>());
        register_handler(EffectPrimitive::COST_REFERENCE, std::make_unique<CostHandler>());
        register_handler(EffectPrimitive::MOVE_TO_UNDER_CARD, std::make_unique<MoveToUnderCardHandler>());
        register_handler(EffectPrimitive::REVEAL_CARDS, std::make_unique<RevealHandler>());
        register_handler(EffectPrimitive::SELECT_NUMBER, std::make_unique<SelectNumberHandler>());
        register_handler(EffectPrimitive::FRIEND_BURST, std::make_unique<FriendBurstHandler>());
        register_handler(EffectPrimitive::GRANT_KEYWORD, std::make_unique<GrantKeywordHandler>());
        register_handler(EffectPrimitive::MOVE_CARD, std::make_unique<MoveCardHandler>());
        register_handler(EffectPrimitive::CAST_SPELL, std::make_unique<CastSpellHandler>());
        register_handler(EffectPrimitive::PUT_CREATURE, std::make_unique<PutCreatureHandler>());
        register_handler(EffectPrimitive::APPLY_MODIFIER, std::make_unique<ModifierHandler>());
        register_handler(EffectPrimitive::SELECT_OPTION, std::make_unique<SelectOptionHandler>());
        register_handler(EffectPrimitive::BREAK_SHIELD, std::make_unique<BreakShieldHandler>());
        register_handler(EffectPrimitive::DISCARD, std::make_unique<DiscardHandler>());
        register_handler(EffectPrimitive::PLAY_FROM_ZONE, std::make_unique<PlayHandler>());
        register_handler(EffectPrimitive::MODIFY_POWER, std::make_unique<ModifyPowerHandler>());
        register_handler(EffectPrimitive::RESOLVE_BATTLE, std::make_unique<ResolveBattleHandler>());

        ConditionSystem::instance().initialize_defaults();

        initialized = true;
    }

    void EffectSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        initialize();
        std::map<std::string, int> empty_context;
        resolve_effect_with_context(game_state, effect, source_instance_id, empty_context, card_db);
    }

    void EffectSystem::resolve_effect_with_context(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        initialize();

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db, execution_context)) {
            return;
        }

        // Full Pipeline Integration: Compile effect and execute once
        std::vector<dm::core::Instruction> instructions;
        compile_effect(game_state, effect, source_instance_id, execution_context, card_db, instructions);

        if (!instructions.empty()) {
            // ResolutionContext is used here primarily to pass params to execute_pipeline
            ResolutionContext ctx(game_state, ActionDef(), source_instance_id, execution_context, card_db);
            execute_pipeline(ctx, instructions);
        }
    }

    void EffectSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context) {
        initialize();

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db, execution_context)) return;

        // Legacy: Just iterate actions.
        for (size_t i = 0; i < effect.actions.size(); ++i) {
            const auto& action = effect.actions[i];

            if (IActionHandler* handler = get_handler(action.type)) {
                ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, &targets);
                handler->resolve_with_targets(ctx);
                continue;
            } else {
                resolve_action(game_state, action, source_instance_id, execution_context, card_db);
            }
        }
    }

    void EffectSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, bool* interrupted, const std::vector<ActionDef>* remaining_actions) {
         initialize();

         // Condition check removed: now handled via compile_action (IF instruction) or check_condition

        // Updated Flow: Always attempt pipeline compilation first
        std::vector<Instruction> pipeline_instructions;
        compile_action(game_state, action, source_instance_id, execution_context, card_db, pipeline_instructions);

        if (!pipeline_instructions.empty()) {
             ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, interrupted, remaining_actions);
             execute_pipeline(ctx, pipeline_instructions);
        }
        else {
             // Check condition manually only if we fallback to handler that relies on it (legacy check)
             // However, strictly speaking, handler->resolve should handle it?
             // Or we rely on check_condition here if pipeline compilation failed?
             // If compilation fails, it means we don't have instructions.
             // If we fallback to `handler->resolve`, we should check condition first if handler doesn't check it.
             // But existing handlers mostly don't check condition inside resolve.
             if (action.condition.has_value()) {
                 if (!check_condition(game_state, *action.condition, source_instance_id, card_db, execution_context)) {
                     return;
                 }
             }

             // Fallback to resolve/resolve_with_targets (which might implement their own pipeline)
             if (IActionHandler* handler = get_handler(action.type)) {
                ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, interrupted, remaining_actions);
                handler->resolve(ctx);
             }
        }
    }

    void EffectSystem::compile_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::vector<dm::core::Instruction>& out_instructions) {
         initialize();

         if (effect.condition.type != "NONE") {
             Instruction if_inst;
             if_inst.op = InstructionOp::IF;
             if_inst.args["cond"] = ConditionSystem::instance().compile_condition(effect.condition);

             std::vector<Instruction> then_block;

             // Compile Actions
             for (const auto& action : effect.actions) {
                 compile_action(game_state, action, source_instance_id, execution_context, card_db, then_block);
             }

             // Compile Commands
             for (const auto& cmd : effect.commands) {
                 nlohmann::json args;
                 args["type"] = "EXECUTE_COMMAND";
                 nlohmann::json cmd_json;
                 dm::core::to_json(cmd_json, cmd);
                 args["cmd"] = cmd_json;
                 then_block.emplace_back(InstructionOp::GAME_ACTION, args);
             }

             if_inst.then_block = then_block;
             out_instructions.push_back(if_inst);
         } else {
             // Compile Actions
             for (const auto& action : effect.actions) {
                 compile_action(game_state, action, source_instance_id, execution_context, card_db, out_instructions);
             }
             // Compile Commands
             for (const auto& cmd : effect.commands) {
                 nlohmann::json args;
                 args["type"] = "EXECUTE_COMMAND";
                 nlohmann::json cmd_json;
                 dm::core::to_json(cmd_json, cmd);
                 args["cmd"] = cmd_json;
                 out_instructions.emplace_back(InstructionOp::GAME_ACTION, args);
             }
         }
    }

    void EffectSystem::compile_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::vector<dm::core::Instruction>& out_instructions) {
        initialize();

        // 0. Handle Action Condition via IF instruction wrapper
        std::vector<dm::core::Instruction> inner_instructions;
        std::vector<dm::core::Instruction>* target_instructions = &out_instructions;

        bool has_condition = action.condition.has_value() && action.condition->type != "NONE";

        if (has_condition) {
            target_instructions = &inner_instructions;
        }

        std::string selection_var_name;

        // 1. Target Selection Scope
        if (action.scope == TargetScope::TARGET_SELECT) {
             Instruction select_inst;
             select_inst.op = InstructionOp::SELECT;

             select_inst.args["filter"] = action.filter;
             select_inst.args["count"] = action.filter.count.value_or(1);

             std::string out_key = "$selection_" + std::to_string(target_instructions->size());
             if (!action.output_value_key.empty()) {
                 out_key = action.output_value_key;
             }
             select_inst.args["out"] = out_key;

             target_instructions->push_back(select_inst);
             selection_var_name = out_key;
        }

        if (IActionHandler* handler = get_handler(action.type)) {
            // Pass selection_var_name to handler via ResolutionContext
            ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, nullptr, nullptr, target_instructions, selection_var_name);
            handler->compile_action(ctx);
        }

        if (has_condition) {
             if (!inner_instructions.empty()) {
                 Instruction if_inst;
                 if_inst.op = InstructionOp::IF;
                 if_inst.args["cond"] = ConditionSystem::instance().compile_condition(*action.condition);
                 if_inst.then_block = inner_instructions;
                 out_instructions.push_back(if_inst);
             }
        }
    }

    void EffectSystem::execute_pipeline(const ResolutionContext& ctx, const std::vector<dm::core::Instruction>& instructions) {
         if (instructions.empty()) return;

         auto pipeline = std::static_pointer_cast<dm::engine::systems::PipelineExecutor>(ctx.game_state.active_pipeline);
         bool is_nested = (pipeline != nullptr);

         // Variables to restore after nested execution
         dm::engine::systems::ContextValue old_source = 0;
         dm::engine::systems::ContextValue old_controller = 0;
         bool has_old_source = false;
         bool has_old_controller = false;

         if (!pipeline) {
             // Root execution: Create new pipeline
             pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();
             ctx.game_state.active_pipeline = pipeline;

             // Populate pipeline context
             for (const auto& kv : ctx.execution_vars) {
                 pipeline->set_context_var(kv.first, kv.second);
             }
             pipeline->set_context_var("$source", ctx.source_instance_id);

             int controller = 0;
             if(ctx.source_instance_id >= 0 && (size_t)ctx.source_instance_id < ctx.game_state.card_owner_map.size()) {
                 controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
             }
             pipeline->set_context_var("$controller", controller);

         } else {
             // Nested execution: Reuse pipeline
             // Save current context values
             auto src_val = pipeline->get_context_var("$source");
             if (!std::holds_alternative<int>(src_val) || std::get<int>(src_val) != 0) { // Check if valid
                 old_source = src_val;
                 has_old_source = true;
             }

             auto ctrl_val = pipeline->get_context_var("$controller");
             if (!std::holds_alternative<int>(ctrl_val) || std::get<int>(ctrl_val) != 0) {
                 old_controller = ctrl_val;
                 has_old_controller = true;
             }

             // Update context
             pipeline->set_context_var("$source", ctx.source_instance_id);

             int controller = 0;
             if(ctx.source_instance_id >= 0 && (size_t)ctx.source_instance_id < ctx.game_state.card_owner_map.size()) {
                 controller = ctx.game_state.card_owner_map[ctx.source_instance_id];
             }
             pipeline->set_context_var("$controller", controller);
         }

         pipeline->execute(instructions, ctx.game_state, ctx.card_db);

         if (!is_nested) {
             // Root execution: Sync back and cleanup
             // Only if the pipeline is finished or paused at root level.
             // If paused, we keep active_pipeline.

             // Sync back execution vars to context (legacy support)
             const auto& pipe_ctx = pipeline->context;
             for (const auto& kv : pipe_ctx) {
                 if (std::holds_alternative<int>(kv.second)) {
                     // Update if key exists or add new?
                     // Legacy code uses [] operator so it adds.
                     ctx.execution_vars[kv.first] = std::get<int>(kv.second);
                 }
             }

             if (pipeline->call_stack.empty()) {
                 ctx.game_state.active_pipeline.reset();
             }
         } else {
             // Nested execution: Restore context
             if (has_old_source) pipeline->set_context_var("$source", old_source);
             if (has_old_controller) pipeline->set_context_var("$controller", old_controller);
         }
    }

    bool EffectSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) {
        if (condition.type == "NONE") return true;

        initialize();
        ConditionSystem& sys = ConditionSystem::instance();
        if (IConditionEvaluator* evaluator = sys.get_evaluator(condition.type)) {
            return evaluator->evaluate(game_state, condition, source_instance_id, card_db, execution_context);
        }

        return true;
    }

    PlayerID EffectSystem::get_controller(const GameState& game_state, int instance_id) {
        const CardInstance* card = game_state.get_card_instance(instance_id);
        if (card) {
            return card->owner;
        }

        if (instance_id >= 0 && instance_id < (int)game_state.card_owner_map.size()) {
            return game_state.card_owner_map[instance_id];
        }
        return game_state.active_player_id;
    }

}
