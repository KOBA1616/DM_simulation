#include "../../effects/effect_resolver.hpp"
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
#include "handlers/friend_burst_handler.hpp" // Added
#include <algorithm>
#include <iostream>
#include <set>

namespace dm::engine {

    using namespace dm::core;

    // Helper to find card instance
    static CardInstance* find_instance(GameState& game_state, int instance_id) {
        for (auto& p : game_state.players) {
            for (auto& c : p.battle_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.hand) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.mana_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.shield_zone) if (c.instance_id == instance_id) return &c;
            for (auto& c : p.graveyard) if (c.instance_id == instance_id) return &c;
        }
        // Also check effect buffer
        for (auto& c : game_state.effect_buffer) if (c.instance_id == instance_id) return &c;

        return nullptr;
    }

    // Helper to determine controller of an instance
    PlayerID GenericCardSystem::get_controller(const GameState& game_state, int instance_id) {
        // Phase A: Use O(1) owner map lookup
        if (instance_id >= 0 && instance_id < (int)game_state.card_owner_map.size()) {
            return game_state.card_owner_map[instance_id];
        }

        // Fallback for instances not in map (should not happen if initialized correctly)
        // or Effect Buffer if they are temp instances?
        // For now, return active player as fallback, but this path should be rare/avoided.
        return game_state.active_player_id;
    }

    // Singleton Registration Helper
    static bool _handlers_registered = false;
    static void ensure_handlers_registered() {
        if (_handlers_registered) return;
        EffectSystem& sys = EffectSystem::instance();
        sys.register_handler(EffectActionType::DRAW_CARD, std::make_unique<DrawHandler>());
        sys.register_handler(EffectActionType::ADD_MANA, std::make_unique<ManaChargeHandler>());
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
        sys.register_handler(EffectActionType::FRIEND_BURST, std::make_unique<FriendBurstHandler>()); // Added
        _handlers_registered = true;
    }

    static bool _evaluators_registered = false;
    static void ensure_evaluators_registered() {
        if (_evaluators_registered) return;
        ConditionSystem& sys = ConditionSystem::instance();
        sys.register_evaluator("DURING_YOUR_TURN", std::make_unique<TurnEvaluator>());
        sys.register_evaluator("DURING_OPPONENT_TURN", std::make_unique<TurnEvaluator>());
        sys.register_evaluator("MANA_ARMED", std::make_unique<ManaArmedEvaluator>());
        sys.register_evaluator("SHIELD_COUNT", std::make_unique<ShieldCountEvaluator>());
        sys.register_evaluator("OPPONENT_PLAYED_WITHOUT_MANA", std::make_unique<OpponentPlayedWithoutManaEvaluator>());
        sys.register_evaluator("CIVILIZATION_MATCH", std::make_unique<CivilizationMatchEvaluator>());
        sys.register_evaluator("FIRST_ATTACK", std::make_unique<FirstAttackEvaluator>());
        _evaluators_registered = true;
    }

    void GenericCardSystem::resolve_trigger(GameState& game_state, TriggerType trigger, int source_instance_id) {
        ensure_handlers_registered();
        ensure_evaluators_registered();
        CardInstance* instance = find_instance(game_state, source_instance_id);
        if (!instance) {
            return;
        }

        std::vector<EffectDef> active_effects;
        const CardData* data = CardRegistry::get_card_data(instance->card_id);
        if (data) {
            active_effects.insert(active_effects.end(), data->effects.begin(), data->effects.end());
            active_effects.insert(active_effects.end(), data->metamorph_abilities.begin(), data->metamorph_abilities.end());
        }

        // Check underlying cards for Metamorph
        for (const auto& under : instance->underlying_cards) {
            const CardData* under_data = CardRegistry::get_card_data(under.card_id);
            if (under_data) {
                active_effects.insert(active_effects.end(), under_data->metamorph_abilities.begin(), under_data->metamorph_abilities.end());
            }
        }

        // bool triggered = false;
        for (const auto& effect : active_effects) {
            if (effect.trigger == trigger) {
                // Stack System: Queue the effect instead of resolving immediately
                PlayerID controller = get_controller(game_state, source_instance_id);
                PendingEffect pending(EffectType::TRIGGER_ABILITY, source_instance_id, controller);
                pending.resolve_type = ResolveType::EFFECT_RESOLUTION; // Handled by ActionGenerator
                pending.effect_def = effect;
                pending.optional = true; // Most triggers are optional? Or check optional flag?
                // Actually EffectDef doesn't have optional flag on top level usually, Actions do.
                // But let's assume players can always choose order, so they are selectable.

                game_state.pending_effects.push_back(pending);
                // triggered = true;
            }
        }
    }

    void GenericCardSystem::resolve_effect(GameState& game_state, const EffectDef& effect, int source_instance_id) {
        ensure_handlers_registered();
        ensure_evaluators_registered();
        std::map<std::string, int> empty_context;
        resolve_effect_with_context(game_state, effect, source_instance_id, empty_context);
    }

    void GenericCardSystem::resolve_effect_with_context(GameState& game_state, const EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context) {
        if (!check_condition(game_state, effect.condition, source_instance_id)) {
            return;
        }

        // Pass context by reference to allow updates
        for (const auto& action : effect.actions) {
            resolve_action(game_state, action, source_instance_id, execution_context);
        }
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
         std::map<std::string, int> empty_context;
         resolve_effect_with_targets(game_state, effect, targets, source_instance_id, card_db, empty_context);
    }

    void GenericCardSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context) {
        ensure_handlers_registered();
        ensure_evaluators_registered();

        if (!check_condition(game_state, effect.condition, source_instance_id)) return;

        for (const auto& action : effect.actions) {

            // Delegate to Handlers for "with targets" execution
            EffectSystem& sys = EffectSystem::instance();
            if (IActionHandler* handler = sys.get_handler(action.type)) {
                handler->resolve_with_targets(game_state, action, targets, source_instance_id, execution_context, card_db);
                continue; // Skip the monolithic block for handled actions
            }

            // Fallback for actions not yet fully migrated or needing legacy logic
            if (action.scope == TargetScope::TARGET_SELECT || action.target_choice == "SELECT") {
                 // Handled by handlers mostly
            } else {
                resolve_action(game_state, action, source_instance_id, execution_context);
            }
        }
    }

    bool GenericCardSystem::check_condition(GameState& game_state, const ConditionDef& condition, int source_instance_id) {
        if (condition.type == "NONE") return true;

        ensure_evaluators_registered();
        ConditionSystem& sys = ConditionSystem::instance();
        if (IConditionEvaluator* evaluator = sys.get_evaluator(condition.type)) {
            return evaluator->evaluate(game_state, condition, source_instance_id);
        }

        // Fallback for unregistered conditions (if any)
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

        // Variable Linking: Override count if input_value_key is present in context
        if (!action.input_value_key.empty()) {
            if (execution_context.count(action.input_value_key)) {
                pending.num_targets_needed = execution_context[action.input_value_key];
            }
        }

        pending.optional = action.optional;
        pending.effect_def = continuation;
        pending.execution_context = execution_context; // Save context

        game_state.pending_effects.push_back(pending);

        return {};
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        std::map<std::string, int> empty;
        resolve_action(game_state, action, source_instance_id, empty);
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context) {
        ensure_handlers_registered();

        // Delegate to Handlers
        EffectSystem& sys = EffectSystem::instance();
        if (IActionHandler* handler = sys.get_handler(action.type)) {
            handler->resolve(game_state, action, source_instance_id, execution_context);
            return;
        }

        // Phase B: GenericCardSystem is now a pure dispatcher.
        // If no handler is found, we do nothing.
        // Target selection must be handled by the specific handler.
    }

}
