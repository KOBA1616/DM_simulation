#include "effect_system.hpp"
#include "card_registry.hpp"
#include "target_utils.hpp"
#include "condition_system.hpp"
#include "engine/infrastructure/commands/command_system.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    void EffectSystem::initialize() {
        if (initialized) return;
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
        
        std::cerr << "[EffectSystem::resolve_effect_with_context] CALLED: commands.size=" << effect.commands.size() << std::endl;

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db, execution_context)) {
            std::cerr << "[EffectSystem::resolve_effect_with_context] Condition check failed, returning" << std::endl;
            return;
        }

        // Use CommandSystem for all commands
        PlayerID controller = get_controller(game_state, source_instance_id);
        for (const auto& cmd : effect.commands) {
            systems::CommandSystem::execute_command(game_state, cmd, source_instance_id, controller, execution_context);
        }
    }

    void EffectSystem::resolve_effect_with_targets(GameState& game_state, const EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context) {
        initialize();

        if (!check_condition(game_state, effect.condition, source_instance_id, card_db, execution_context)) return;

        // CommandDef based processing (New System)
        // Store targets in execution_context for CommandSystem to use
        if (!targets.empty()) {
            execution_context["$targets_count"] = static_cast<int>(targets.size());
            for (size_t i = 0; i < targets.size(); ++i) {
                execution_context["$target_" + std::to_string(i)] = targets[i];
            }
        }

        // Use CommandSystem for all commands
        PlayerID controller = get_controller(game_state, source_instance_id);
        for (const auto& cmd : effect.commands) {
            systems::CommandSystem::execute_command(game_state, cmd, source_instance_id, controller, execution_context);
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

        if (instance_id >= 0) {
            return game_state.get_card_owner(instance_id);
        }
        return game_state.active_player_id;
    }

}
