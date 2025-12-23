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

        register_handler(EffectActionType::DRAW_CARD, std::make_unique<DrawHandler>());
        register_handler(EffectActionType::ADD_MANA, std::make_unique<ManaChargeHandler>());
        register_handler(EffectActionType::SEND_TO_MANA, std::make_unique<ManaChargeHandler>());
        register_handler(EffectActionType::DESTROY, std::make_unique<DestroyHandler>());
        register_handler(EffectActionType::RETURN_TO_HAND, std::make_unique<ReturnToHandHandler>());
        register_handler(EffectActionType::TAP, std::make_unique<TapHandler>());
        register_handler(EffectActionType::UNTAP, std::make_unique<UntapHandler>());
        register_handler(EffectActionType::COUNT_CARDS, std::make_unique<CountHandler>());
        register_handler(EffectActionType::GET_GAME_STAT, std::make_unique<CountHandler>());
        register_handler(EffectActionType::ADD_SHIELD, std::make_unique<ShieldHandler>());
        register_handler(EffectActionType::SEND_SHIELD_TO_GRAVE, std::make_unique<ShieldHandler>());
        register_handler(EffectActionType::SEARCH_DECK, std::make_unique<SearchHandler>());
        register_handler(EffectActionType::SEARCH_DECK_BOTTOM, std::make_unique<SearchHandler>());
        register_handler(EffectActionType::SEND_TO_DECK_BOTTOM, std::make_unique<SearchHandler>());
        register_handler(EffectActionType::SHUFFLE_DECK, std::make_unique<SearchHandler>());
        register_handler(EffectActionType::MEKRAID, std::make_unique<BufferHandler>());
        register_handler(EffectActionType::LOOK_TO_BUFFER, std::make_unique<BufferHandler>());
        register_handler(EffectActionType::MOVE_BUFFER_TO_ZONE, std::make_unique<BufferHandler>());
        register_handler(EffectActionType::PLAY_FROM_BUFFER, std::make_unique<BufferHandler>());
        register_handler(EffectActionType::COST_REFERENCE, std::make_unique<CostHandler>());
        register_handler(EffectActionType::MOVE_TO_UNDER_CARD, std::make_unique<MoveToUnderCardHandler>());
        register_handler(EffectActionType::REVEAL_CARDS, std::make_unique<RevealHandler>());
        register_handler(EffectActionType::SELECT_NUMBER, std::make_unique<SelectNumberHandler>());
        register_handler(EffectActionType::FRIEND_BURST, std::make_unique<FriendBurstHandler>());
        register_handler(EffectActionType::GRANT_KEYWORD, std::make_unique<GrantKeywordHandler>());
        register_handler(EffectActionType::MOVE_CARD, std::make_unique<MoveCardHandler>());
        register_handler(EffectActionType::CAST_SPELL, std::make_unique<CastSpellHandler>());
        register_handler(EffectActionType::PUT_CREATURE, std::make_unique<PutCreatureHandler>());
        register_handler(EffectActionType::APPLY_MODIFIER, std::make_unique<ModifierHandler>());
        register_handler(EffectActionType::SELECT_OPTION, std::make_unique<SelectOptionHandler>());
        register_handler(EffectActionType::BREAK_SHIELD, std::make_unique<BreakShieldHandler>());
        register_handler(EffectActionType::DISCARD, std::make_unique<DiscardHandler>());
        register_handler(EffectActionType::PLAY_FROM_ZONE, std::make_unique<PlayHandler>());
        register_handler(EffectActionType::MODIFY_POWER, std::make_unique<ModifyPowerHandler>());
        register_handler(EffectActionType::RESOLVE_BATTLE, std::make_unique<ResolveBattleHandler>());

        ConditionSystem::instance().initialize_defaults();

        initialized = true;
    }

    void EffectSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        initialize();
        CardInstance* instance = game_state.get_card_instance(source_instance_id);
        if (!instance) {
            return;
        }

        std::vector<EffectDef> active_effects;
        bool found = false;

        const CardDefinition* def_ptr = nullptr;
        if (card_db.count(instance->card_id)) {
            def_ptr = &card_db.at(instance->card_id);
            found = true;
        } else if (CardRegistry::get_all_definitions().count(instance->card_id)) {
            def_ptr = &CardRegistry::get_all_definitions().at(instance->card_id);
        }

        if (def_ptr) {
            const auto& data = *def_ptr;
            active_effects.insert(active_effects.end(), data.effects.begin(), data.effects.end());
            active_effects.insert(active_effects.end(), data.metamorph_abilities.begin(), data.metamorph_abilities.end());

            if (data.keywords.friend_burst && trigger == TriggerType::ON_PLAY) {
                EffectDef fb_effect;
                fb_effect.trigger = TriggerType::ON_PLAY;

                ActionDef fb_action;
                fb_action.type = EffectActionType::FRIEND_BURST;
                fb_action.scope = TargetScope::TARGET_SELECT;
                fb_action.optional = true;
                fb_action.filter.owner = "SELF";
                fb_action.filter.zones = {"BATTLE_ZONE"};
                fb_action.filter.types = {"CREATURE"};
                fb_action.filter.is_tapped = false;
                fb_action.filter.count = 1;

                fb_effect.actions.push_back(fb_action);
                active_effects.push_back(fb_effect);
            }
        }

        for (const auto& under : instance->underlying_cards) {
            if (card_db.count(under.card_id)) {
                const auto& under_data = card_db.at(under.card_id);
                active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
            } else if (CardRegistry::get_all_definitions().count(under.card_id)) {
                const auto& under_data = CardRegistry::get_all_definitions().at(under.card_id);
                active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
            }
        }

        PlayerID controller = get_controller(game_state, source_instance_id);

        for (const auto& effect : active_effects) {
            if (effect.trigger == trigger) {
                PendingEffect pending(EffectType::TRIGGER_ABILITY, source_instance_id, controller);
                pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                pending.effect_def = effect;
                pending.optional = true;
                pending.chain_depth = game_state.turn_stats.current_chain_depth + 1;
                game_state.pending_effects.push_back(pending);
            }
        }
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

        // Note: For target handling, we might need to inject targets into the pipeline context
        // Currently resolve_effect_with_targets is primarily used for handling selected targets.
        // We can inject them as $selection or similar.

        // This method is often called AFTER select_targets returns.
        // If we switch to full pipeline, the selection logic should ideally be inside the pipeline via SELECT instruction.
        // However, existing code flow (PendingEffect -> Target Selection -> resolve_effect_with_targets) persists.

        // Strategy: Pre-populate context with targets, then run pipeline.
        // But compile_effect generates target selection instructions if action has scope.
        // If we already selected targets, we shouldn't re-select.
        // The calling code (PendingEffectStrategy) usually clears the scope in the effect passed here?
        // Or we set a flag.

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

         if (action.condition.has_value()) {
            bool condition_met = false;
            if (action.condition->type == "COMPARE_STAT" && execution_context.count(action.condition->stat_key)) {
                int val = execution_context[action.condition->stat_key];
                int target = action.condition->value;
                std::string op = action.condition->op;
                if (op == ">") condition_met = val > target;
                else if (op == ">=") condition_met = val >= target;
                else if (op == "<") condition_met = val < target;
                else if (op == "<=") condition_met = val <= target;
                else if (op == "=" || op == "==") condition_met = val == target;
                else if (op == "!=") condition_met = val != target;
            } else {
                 condition_met = check_condition(game_state, *action.condition, source_instance_id, card_db, execution_context);
            }

            if (!condition_met) return;
        }

        // Updated Flow: Always attempt pipeline compilation first
        std::vector<Instruction> pipeline_instructions;
        compile_action(game_state, action, source_instance_id, execution_context, card_db, pipeline_instructions);

        if (!pipeline_instructions.empty()) {
             ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, interrupted, remaining_actions);
             execute_pipeline(ctx, pipeline_instructions);
        }
        else {
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
             nlohmann::json cond_json;
             cond_json["type"] = effect.condition.type;
             cond_json["value"] = effect.condition.value;
             cond_json["str_val"] = effect.condition.str_val;
             if_inst.args["cond"] = cond_json;

             std::vector<Instruction> then_block;

             // Compile Actions
             for (const auto& action : effect.actions) {
                 compile_action(game_state, action, source_instance_id, execution_context, card_db, then_block);
             }

             // Compile Commands (Fix: Actually generate EXECUTE_COMMAND instructions)
             for (const auto& cmd : effect.commands) {
                 nlohmann::json args;
                 args["type"] = "EXECUTE_COMMAND";
                 // Serialize CommandDef to JSON
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

        std::string selection_var_name;

        // 1. Target Selection Scope
        if (action.scope == TargetScope::TARGET_SELECT) {
             Instruction select_inst;
             select_inst.op = InstructionOp::SELECT;

             select_inst.args["filter"] = action.filter;
             select_inst.args["count"] = action.filter.count.value_or(1);

             std::string out_key = "$selection_" + std::to_string(out_instructions.size());
             if (!action.output_value_key.empty()) {
                 out_key = action.output_value_key;
             }
             select_inst.args["out"] = out_key;

             out_instructions.push_back(select_inst);
             selection_var_name = out_key;
        }

        if (IActionHandler* handler = get_handler(action.type)) {
            // Pass selection_var_name to handler via ResolutionContext
            ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, nullptr, nullptr, &out_instructions, selection_var_name);
            handler->compile_action(ctx);
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

        PlayerID controller = get_controller(game_state, source_instance_id);
        if (condition.type == "DURING_YOUR_TURN") {
            return game_state.active_player_id == controller;
        }
        if (condition.type == "DURING_OPPONENT_TURN") {
            return game_state.active_player_id != controller;
        }

        return true;
    }

    std::vector<int> EffectSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id, const EffectDef& continuation, std::map<std::string, int>& execution_context) {
        PlayerID controller = get_controller(game_state, source_instance_id);

        PendingEffect pending(EffectType::NONE, source_instance_id, controller);
        pending.resolve_type = ResolveType::TARGET_SELECT;
        pending.filter = action.filter;

        if (pending.filter.zones.empty()) {
             if (action.target_choice == "ALL_ENEMY") {
                 pending.filter.owner = "OPPONENT";
                 pending.filter.zones = {"BATTLE_ZONE"};
             }
        }

        if (action.filter.count.has_value()) {
            pending.num_targets_needed = action.filter.count.value();
        } else {
            pending.num_targets_needed = 1;
        }

        if (!action.input_value_key.empty()) {
            if (execution_context.count(action.input_value_key)) {
                pending.num_targets_needed = execution_context[action.input_value_key];
            }
        }

        pending.optional = action.optional;
        pending.effect_def = continuation;
        pending.execution_context = execution_context;

        game_state.pending_effects.push_back(pending);

        return {};
    }

    void EffectSystem::delegate_selection(const ResolutionContext& ctx) {
        if (!ctx.interrupted) return;

        dm::core::EffectDef ed;
        ed.trigger = dm::core::TriggerType::NONE;
        ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
        ed.actions = { ctx.action };
        if (ctx.remaining_actions) {
            ed.actions.insert(ed.actions.end(), ctx.remaining_actions->begin(), ctx.remaining_actions->end());
        }

        instance().select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
        *ctx.interrupted = true;
    }

    void EffectSystem::check_mega_last_burst(dm::core::GameState& game_state, const dm::core::CardInstance& card, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         const CardDefinition* def_ptr = nullptr;
         if (card_db.count(card.card_id)) {
             def_ptr = &card_db.at(card.card_id);
         } else if (CardRegistry::get_all_definitions().count(card.card_id)) {
             def_ptr = &CardRegistry::get_all_definitions().at(card.card_id);
         }

         if (!def_ptr) return;
         const auto& def = *def_ptr;

         if (def.keywords.mega_last_burst && def.spell_side) {
             PlayerID controller = get_controller(game_state, card.instance_id);

             EffectDef eff;
             eff.trigger = TriggerType::NONE;

             ActionDef act;
             act.type = EffectActionType::CAST_SPELL;
             act.scope = TargetScope::TARGET_SELECT;
             act.optional = true;
             act.cast_spell_side = true;

             eff.actions.push_back(act);

             PendingEffect pending(EffectType::TRIGGER_ABILITY, card.instance_id, controller);
             pending.resolve_type = ResolveType::TARGET_SELECT;
             pending.target_instance_ids.push_back(card.instance_id);
             pending.num_targets_needed = 1;
             pending.effect_def = eff;
             pending.optional = true;

             game_state.pending_effects.push_back(pending);
         }
    }

    PlayerID EffectSystem::get_controller(const GameState& game_state, int instance_id) {
        if (instance_id >= 0 && instance_id < (int)game_state.card_owner_map.size()) {
            return game_state.card_owner_map[instance_id];
        }
        return game_state.active_player_id;
    }

}
