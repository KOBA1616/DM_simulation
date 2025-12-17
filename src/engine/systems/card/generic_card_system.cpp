#include "engine/effects/effect_resolver.hpp"
#include "generic_card_system.hpp"
#include "card_registry.hpp"
#include "target_utils.hpp"
#include "effect_system.hpp"
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
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/command_system.hpp"
#include "legacy_converter.hpp"
#include <algorithm>
#include <iostream>
#include <set>

namespace dm::engine {

    using namespace dm::engine::systems; // For PipelineExecutor

    using namespace dm::core;

    // Helper to find card instance
    static CardInstance* find_instance(GameState& game_state, int instance_id) {
        for (auto& p : game_state.players) {
            for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
            // Also check effect buffer per player
            for (auto& c : p.effect_buffer) if (c.instance_id == instance_id) return &c;
        }
        return nullptr;
    }

    // Helper to determine controller of an instance
    PlayerID GenericCardSystem::get_controller(const GameState& game_state, int instance_id) {
        if (instance_id >= 0 && instance_id < (int)game_state.card_owner_map.size()) {
            return game_state.card_owner_map[instance_id];
        }
        return game_state.active_player_id;
    }

    // Singleton Registration Helper
    static bool _handlers_registered = false;
    static void ensure_handlers_registered() {
        if (_handlers_registered) return;
        EffectSystem& sys = EffectSystem::instance();
        sys.register_handler(EffectActionType::DRAW_CARD, std::make_unique<DrawHandler>());
        sys.register_handler(EffectActionType::ADD_MANA, std::make_unique<ManaChargeHandler>());
        sys.register_handler(EffectActionType::SEND_TO_MANA, std::make_unique<ManaChargeHandler>());
        sys.register_handler(EffectActionType::DESTROY, std::make_unique<DestroyHandler>());
        sys.register_handler(EffectActionType::RETURN_TO_HAND, std::make_unique<ReturnToHandHandler>());
        sys.register_handler(EffectActionType::TAP, std::make_unique<TapHandler>());
        sys.register_handler(EffectActionType::UNTAP, std::make_unique<UntapHandler>());
        sys.register_handler(EffectActionType::COUNT_CARDS, std::make_unique<CountHandler>());
        sys.register_handler(EffectActionType::GET_GAME_STAT, std::make_unique<CountHandler>());
        sys.register_handler(EffectActionType::ADD_SHIELD, std::make_unique<ShieldHandler>());
        sys.register_handler(EffectActionType::SEND_SHIELD_TO_GRAVE, std::make_unique<ShieldHandler>());
        sys.register_handler(EffectActionType::SEARCH_DECK, std::make_unique<SearchHandler>());
        sys.register_handler(EffectActionType::SEARCH_DECK_BOTTOM, std::make_unique<SearchHandler>());
        sys.register_handler(EffectActionType::SEND_TO_DECK_BOTTOM, std::make_unique<SearchHandler>());
        sys.register_handler(EffectActionType::SHUFFLE_DECK, std::make_unique<SearchHandler>());
        sys.register_handler(EffectActionType::MEKRAID, std::make_unique<BufferHandler>());
        sys.register_handler(EffectActionType::LOOK_TO_BUFFER, std::make_unique<BufferHandler>());
        sys.register_handler(EffectActionType::MOVE_BUFFER_TO_ZONE, std::make_unique<BufferHandler>());
        sys.register_handler(EffectActionType::PLAY_FROM_BUFFER, std::make_unique<BufferHandler>());
        sys.register_handler(EffectActionType::COST_REFERENCE, std::make_unique<CostHandler>());
        sys.register_handler(EffectActionType::MOVE_TO_UNDER_CARD, std::make_unique<MoveToUnderCardHandler>());
        sys.register_handler(EffectActionType::REVEAL_CARDS, std::make_unique<RevealHandler>());
        sys.register_handler(EffectActionType::SELECT_NUMBER, std::make_unique<SelectNumberHandler>());
        sys.register_handler(EffectActionType::FRIEND_BURST, std::make_unique<FriendBurstHandler>());
        sys.register_handler(EffectActionType::GRANT_KEYWORD, std::make_unique<GrantKeywordHandler>());
        sys.register_handler(EffectActionType::MOVE_CARD, std::make_unique<MoveCardHandler>());
        sys.register_handler(EffectActionType::CAST_SPELL, std::make_unique<CastSpellHandler>());
        sys.register_handler(EffectActionType::PUT_CREATURE, std::make_unique<PutCreatureHandler>());
        sys.register_handler(EffectActionType::APPLY_MODIFIER, std::make_unique<ModifierHandler>());
        sys.register_handler(EffectActionType::SELECT_OPTION, std::make_unique<SelectOptionHandler>());
        sys.register_handler(EffectActionType::BREAK_SHIELD, std::make_unique<BreakShieldHandler>());
        sys.register_handler(EffectActionType::DISCARD, std::make_unique<DiscardHandler>());
        sys.register_handler(EffectActionType::PLAY_FROM_ZONE, std::make_unique<PlayHandler>());
        sys.register_handler(EffectActionType::MODIFY_POWER, std::make_unique<ModifyPowerHandler>());
        _handlers_registered = true;
    }

    static bool _evaluators_registered = false;
    static void ensure_evaluators_registered() {
        if (_evaluators_registered) return;
        ConditionSystem::instance().initialize_defaults();
        _evaluators_registered = true;
    }

    void GenericCardSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        ensure_handlers_registered();
        ensure_evaluators_registered();
        CardInstance* instance = find_instance(game_state, source_instance_id);
        if (!instance) {
            std::cerr << "resolve_trigger: Instance " << source_instance_id << " not found." << std::endl;
            return;
        }

        std::vector<EffectDef> active_effects;
        bool found = false;
        if (card_db.count(instance->card_id)) {
            const auto& data = card_db.at(instance->card_id);
            active_effects.insert(active_effects.end(), data.effects.begin(), data.effects.end());
            active_effects.insert(active_effects.end(), data.metamorph_abilities.begin(), data.metamorph_abilities.end());

            // Friend Burst Logic
            if (data.keywords.friend_burst && trigger == TriggerType::ON_PLAY) {
                EffectDef fb_effect;
                fb_effect.trigger = TriggerType::ON_PLAY;

                ActionDef fb_action;
                fb_action.type = EffectActionType::FRIEND_BURST;
                fb_action.scope = TargetScope::TARGET_SELECT;
                fb_action.optional = true; // "You may tap..."

                fb_action.filter.owner = "SELF";
                fb_action.filter.zones = {"BATTLE_ZONE"};
                fb_action.filter.types = {"CREATURE"};
                fb_action.filter.is_tapped = false; // Must tap an untapped creature
                fb_action.filter.count = 1;

                // Note: Friend Burst requires tapping *another* creature.
                // Target filtering for "exclude self" is not fully implicit in FilterDef yet,
                // but standard play logic handles this via UI or validation.

                fb_effect.actions.push_back(fb_action);
                active_effects.push_back(fb_effect);
            }

            found = true;
        }

        if (!found || active_effects.empty()) {
            const auto& registry = CardRegistry::get_all_definitions();
            if (registry.count(instance->card_id)) {
                 const auto& data = registry.at(instance->card_id);
                 active_effects.insert(active_effects.end(), data.effects.begin(), data.effects.end());
                 active_effects.insert(active_effects.end(), data.metamorph_abilities.begin(), data.metamorph_abilities.end());
                 // std::cerr << "resolve_trigger: Fallback to registry for ID " << instance->card_id << ". Effects: " << data.effects.size() << std::endl;
            } else {
                 std::cerr << "resolve_trigger: Card ID " << instance->card_id << " not found in DB or Registry." << std::endl;
            }
        }

        for (const auto& under : instance->underlying_cards) {
            bool found_under = false;
            if (card_db.count(under.card_id)) {
                const auto& under_data = card_db.at(under.card_id);
                active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
                found_under = true;
            }
            if (!found_under) {
                const auto& registry = CardRegistry::get_all_definitions();
                if (registry.count(under.card_id)) {
                    const auto& under_data = registry.at(under.card_id);
                    active_effects.insert(active_effects.end(), under_data.metamorph_abilities.begin(), under_data.metamorph_abilities.end());
                }
            }
        }

        for (const auto& effect : active_effects) {
            if (effect.trigger == trigger) {
                PlayerID controller = get_controller(game_state, source_instance_id);
                PendingEffect pending(EffectType::TRIGGER_ABILITY, source_instance_id, controller);
                pending.resolve_type = ResolveType::EFFECT_RESOLUTION;
                pending.effect_def = effect;
                pending.optional = true;

                // Step 5.2.2: Loop Prevention
                pending.chain_depth = game_state.turn_stats.current_chain_depth + 1;

                game_state.pending_effects.push_back(pending);
            }
        }
    }

    void GenericCardSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        ensure_handlers_registered();
        ensure_evaluators_registered();
        std::map<std::string, int> empty_context;
        resolve_effect_with_context(game_state, effect, source_instance_id, empty_context, card_db);
    }

        void GenericCardSystem::resolve_effect_with_context(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db)) {
            return;
        }

        for (size_t i = 0; i < effect.actions.size(); ++i) {
            const auto& action = effect.actions[i];

            // Construct remaining actions for continuation
            std::vector<ActionDef> remaining_actions;
            if (i + 1 < effect.actions.size()) {
                remaining_actions.insert(remaining_actions.end(), effect.actions.begin() + i + 1, effect.actions.end());
            }

            bool interrupted = false;
            resolve_action(game_state, action, source_instance_id, execution_context, card_db, &interrupted, &remaining_actions);

            if (interrupted) {
                // std::cout << "resolve_effect: Interrupted loop." << std::endl;
                break;
            }
        }

        // Phase 7: Hybrid Engine - Execute Commands
        for (const auto& cmd : effect.commands) {
            PlayerID controller = get_controller(game_state, source_instance_id);
            dm::engine::systems::CommandSystem::execute_command(game_state, cmd, source_instance_id, controller);
        }
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         std::map<std::string, int> empty_context;
         resolve_effect_with_targets(game_state, effect, targets, source_instance_id, card_db, empty_context);
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context) {
        ensure_handlers_registered();
        ensure_evaluators_registered();

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db)) return;

        for (size_t i = 0; i < effect.actions.size(); ++i) {
            const auto& action = effect.actions[i];

            // Delegate to Handlers for "with targets" execution
            EffectSystem& sys = EffectSystem::instance();
            if (IActionHandler* handler = sys.get_handler(action.type)) {
                // Create Context
                ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, &targets);
                handler->resolve_with_targets(ctx);
                continue;
            }

            // Fallback for actions not yet fully migrated or needing legacy logic
            if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                 // Handled by handlers mostly
            } else {
                 // We don't support interruption here for legacy fallback in resolve_effect_with_targets
                resolve_action(game_state, action, source_instance_id, execution_context, card_db);
            }
        }
    }

    bool GenericCardSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) {
        if (condition.type == "NONE") return true;

        ensure_evaluators_registered();
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

    std::vector<int> GenericCardSystem::select_targets(GameState& game_state, const ActionDef& action, int source_instance_id, const EffectDef& continuation, std::map<std::string, int>& execution_context) {
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

        // Variable Linking
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

    void GenericCardSystem::delegate_selection(const ResolutionContext& ctx) {
        if (!ctx.interrupted) {
             std::cerr << "delegate_selection: ctx.interrupted is null!" << std::endl;
             return;
        }

        dm::core::EffectDef ed;
        ed.trigger = dm::core::TriggerType::NONE;
        ed.condition = dm::core::ConditionDef{"NONE", 0, "", "", "", std::nullopt};
        ed.actions = { ctx.action };
        if (ctx.remaining_actions) {
            ed.actions.insert(ed.actions.end(), ctx.remaining_actions->begin(), ctx.remaining_actions->end());
        }

        GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
        if (ctx.interrupted) {
            *ctx.interrupted = true;
            // std::cout << "delegate_selection: Interrupted set to true." << std::endl;
        }
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        std::map<std::string, int> empty;
        // Use Registry as fallback if no DB provided
        resolve_action(game_state, action, source_instance_id, empty, CardRegistry::get_all_definitions());
    }

    // Binding Helper
    void GenericCardSystem::resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         resolve_action(game_state, action, source_instance_id, execution_context, card_db, nullptr, nullptr);
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, bool* interrupted, const std::vector<ActionDef>* remaining_actions) {
        ensure_handlers_registered();

        // Check Action-level condition (if any)
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
                 condition_met = check_condition(game_state, *action.condition, source_instance_id, card_db);
            }

            if (!condition_met) {
                return;
            }
        }

        // Phase 6: Full Trigger System Migration
        // Attempt to convert Legacy Action to CommandDef
        if (auto cmd_opt = dm::engine::systems::LegacyConverter::convert(action)) {
            PlayerID controller = get_controller(game_state, source_instance_id);
            // Execute via CommandSystem
            // Note: CommandSystem handles basic resolution.
            // Interactive targeting (SELECT -> TARGET) is not fully supported in CommandSystem yet for all cases,
            // but LegacyConverter only converts supported types.
            dm::engine::systems::CommandSystem::execute_command(game_state, *cmd_opt, source_instance_id, controller);
            return;
        }

        EffectSystem& sys = EffectSystem::instance();
        if (IActionHandler* handler = sys.get_handler(action.type)) {
            ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db, nullptr, interrupted, remaining_actions);
            handler->resolve(ctx);
            return;
        }
    }

    void GenericCardSystem::check_mega_last_burst(dm::core::GameState& game_state, const dm::core::CardInstance& card, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         const CardDefinition* def_ptr = nullptr;
         if (card_db.count(card.card_id)) {
             def_ptr = &card_db.at(card.card_id);
         } else {
             const auto& registry = CardRegistry::get_all_definitions();
             if (registry.count(card.card_id)) {
                 def_ptr = &registry.at(card.card_id);
             } else {
                 std::cerr << "check_mega_last_burst: Card ID " << card.card_id << " not found in DB or Registry." << std::endl;
             }
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

}
