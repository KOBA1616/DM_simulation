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
        sys.register_evaluator("COMPARE_STAT", std::make_unique<CompareStatEvaluator>());
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

        for (const auto& action : effect.actions) {
            resolve_action(game_state, action, source_instance_id, execution_context, card_db);
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

        for (const auto& action : effect.actions) {

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

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id) {
        std::map<std::string, int> empty;
        std::map<dm::core::CardID, dm::core::CardDefinition> empty_db;
        resolve_action(game_state, action, source_instance_id, empty, empty_db);
    }

    void GenericCardSystem::resolve_action(GameState& game_state, const ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
        ensure_handlers_registered();

        // Check Action-level condition (if any)
        if (action.condition.has_value()) {
            // Pass context to condition checker?
            // Currently check_condition uses GenericCardSystem::check_condition which doesn't take context directly
            // BUT, for COMPARE_STAT, it might need to look up variables.
            // check_condition signature: (GameState, ConditionDef, source_id, card_db)
            // It does NOT currently support context variables.
            // We need to support it if we want to check "destroyed_count" from context.

            // HACK: If the condition is COMPARE_STAT and the stat_key is in execution_context,
            // we manually evaluate it here or extend check_condition.
            // For proper modularity, check_condition should accept context.
            // Let's implement a local check for context variables for now.

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

        EffectSystem& sys = EffectSystem::instance();
        if (IActionHandler* handler = sys.get_handler(action.type)) {
            ResolutionContext ctx(game_state, action, source_instance_id, execution_context, card_db);
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

         // Debug output
         // std::cout << "Checking Mega Last Burst for ID " << card.card_id << ". Keyword: " << def.keywords.mega_last_burst << ", SpellSide: " << (def.spell_side ? "Yes" : "No") << std::endl;

         if (def.keywords.mega_last_burst && def.spell_side) {
             PlayerID controller = get_controller(game_state, card.instance_id); // This might need card_owner_map lookup as card is already moved
             // Since card was moved, get_controller might return active_player if not in map, but card instance usually preserves ID?
             // Actually, get_controller uses card_owner_map which persists across moves.

             // Queue the effect: You may cast spell side.
             EffectDef eff;
             eff.trigger = TriggerType::NONE;

             ActionDef act;
             act.type = EffectActionType::CAST_SPELL;
             act.scope = TargetScope::TARGET_SELECT; // Or SELF?
             // CAST_SPELL handler logic: if we pass explicit targets (the card itself), it moves it to stack and casts.
             // We want to target THIS card in its current zone.
             // But we need to select it?
             // Or we can just set targets manually in PendingEffect.
             act.optional = true; // "You may"
             act.cast_spell_side = true;

             eff.actions.push_back(act);

             PendingEffect pending(EffectType::TRIGGER_ABILITY, card.instance_id, controller);
             pending.resolve_type = ResolveType::EFFECT_RESOLUTION; // We construct effect manually
             pending.effect_def = eff;
             pending.optional = true;

             // IMPORTANT: We must target the card in its NEW zone.
             // But CAST_SPELL handler takes targets via ctx.targets if coming from TARGET_SELECT,
             // OR if we skip selection, we need to provide the target.

             // If we use TARGET_SELECT, the user has to click.
             // "Mega Last Burst" allows user to CHOOSE to cast.
             // So showing a "Use Mega Last Burst?" dialog is appropriate.
             // PendingEffect with optional=true handles the "Yes/No" dialog for the effect itself.

             // Now, for the action. We want to execute CAST_SPELL on THIS card.
             // We can pre-fill the target in the pending effect so no selection is needed?
             // If we set resolve_type = TARGET_SELECT, it asks for target.
             // We want resolve_type = EFFECT_RESOLUTION, executing the action.
             // But CAST_SPELL needs a target.
             // If ActionDef has no filter/scope, it might default to source?
             // Let's modify CAST_SPELL to handle "Self" if no targets?
             // Or better: Create a wrapper effect that has the action.
             // And we inject a wrapper "SELECT_TARGET" action? No.

             // If we use GenericCardSystem::resolve_effect_with_targets, we can pass the target.
             // But that's for immediate resolution.
             // Here we want to queue it.

             // If we queue TriggerType::TRIGGER_ABILITY, GenericCardSystem::resolve_effect is called.
             // Then resolve_action is called.
             // Then CastSpellHandler::resolve is called.
             // CastSpellHandler::resolve DOES NOTHING currently (only resolve_with_targets implemented).

             // So we need to use TARGET_SELECT to feed resolve_with_targets.
             // ActionDef: scope = TARGET_SELECT, filter = { owner: SELF, zones: [CURRENT_ZONE], count: 1 }?
             // And we restrict it to THIS card.
             // That requires a specific filter for ID? FilterDef doesn't support ID.

             // Alternatively, we use `resolve_effect_with_targets` directly if we confirm "Yes".
             // But PendingEffect structure assumes `effect_def` is just run.

             // Workaround:
             // We queue a PendingEffect that, when resolved (user says Yes),
             // executes `resolve_effect_with_targets` with the card ID pre-filled.
             // But `ActionType::RESOLVE_EFFECT` in `EffectResolver` calls `GenericCardSystem::resolve_effect` which doesn't take targets.

             // However, `ActionType::RESOLVE_EFFECT` also handles `ResolveType::TARGET_SELECT`.
             // If we set `pe.resolve_type = ResolveType::TARGET_SELECT` and `pe.target_instance_ids = { card.instance_id }`,
             // then `EffectResolver` calls `resolve_effect_with_targets`.
             // But `TARGET_SELECT` usually implies "Waiting for selection".
             // If we pre-fill `target_instance_ids`, does it auto-resolve?
             // No, `ActionType::RESOLVE_EFFECT` is generated when the stack processes.
             // If `num_targets_needed` is met, `StackStrategy` generates `RESOLVE_EFFECT`.

             // So:
             pending.resolve_type = ResolveType::TARGET_SELECT;
             pending.target_instance_ids.push_back(card.instance_id);
             pending.num_targets_needed = 1;
             pending.effect_def = eff; // Contains CAST_SPELL action

             // But we need to make sure the user sees "Use Mega Last Burst?"
             // `optional = true` on PendingEffect usually prompts the user before generating `SELECT_TARGET`.
             // But here we skip `SELECT_TARGET` generation because targets are full?
             // `StackStrategy` or `PendingEffectStrategy` checks this.
             // If targets are full, it generates RESOLVE_EFFECT immediately.

             // So the flow:
             // 1. Queue PendingEffect (Optional).
             // 2. User gets prompt "Use Mega Last Burst?".
             // 3. If Yes, `RESOLVE_EFFECT` is generated.
             // 4. `RESOLVE_EFFECT` executes `resolve_effect_with_targets` using the pre-filled target.
             // 5. `resolve_effect_with_targets` calls `CastSpellHandler::resolve_with_targets`.
             // 6. Handler uses `cast_spell_side` flag and casts the spell.

             game_state.pending_effects.push_back(pending);
         }
    }

}
