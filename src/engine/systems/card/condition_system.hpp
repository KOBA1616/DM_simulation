#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "generic_card_system.hpp"

namespace dm::engine {

    class IConditionEvaluator {
    public:
        virtual ~IConditionEvaluator() = default;
        virtual bool evaluate(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) = 0;
    };

    class ConditionSystem {
    public:
        static ConditionSystem& instance() {
            static ConditionSystem instance;
            return instance;
        }

        void register_evaluator(const std::string& type, std::unique_ptr<IConditionEvaluator> evaluator) {
            evaluators[type] = std::move(evaluator);
        }

        IConditionEvaluator* get_evaluator(const std::string& type) {
            if (evaluators.count(type)) {
                return evaluators[type].get();
            }
            return nullptr;
        }

        bool evaluate_def(dm::core::GameState& state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context) {
            if (condition.type == "NONE" || condition.type.empty()) return true;
            if (IConditionEvaluator* eval = get_evaluator(condition.type)) {
                return eval->evaluate(state, condition, source_instance_id, card_db, execution_context);
            }
            return false;
        }

        void initialize_defaults();

    private:
        ConditionSystem() = default;
        ConditionSystem(const ConditionSystem&) = delete;
        ConditionSystem& operator=(const ConditionSystem&) = delete;
        std::map<std::string, std::unique_ptr<IConditionEvaluator>> evaluators;
    };
}
