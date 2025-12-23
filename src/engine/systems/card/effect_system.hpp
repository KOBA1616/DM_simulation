#pragma once
#include "core/game_state.hpp"
#include "core/card_json_types.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dm::engine {

    struct ResolutionContext {
        dm::core::GameState& game_state;
        const dm::core::ActionDef& action;
        int source_instance_id;
        std::map<std::string, int>& execution_vars;
        const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db;
        const std::vector<int>* targets = nullptr;
        bool* interrupted = nullptr;
        const std::vector<dm::core::ActionDef>* remaining_actions = nullptr;
        std::vector<dm::core::Instruction>* instruction_buffer = nullptr;
        std::string selection_var; // Added for pipeline dynamic selection

        ResolutionContext(
            dm::core::GameState& state,
            const dm::core::ActionDef& act,
            int src,
            std::map<std::string, int>& vars,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& db,
            const std::vector<int>* tgs = nullptr,
            bool* intr = nullptr,
            const std::vector<dm::core::ActionDef>* rem = nullptr,
            std::vector<dm::core::Instruction>* inst_buf = nullptr,
            std::string sel_var = "")
            : game_state(state), action(act), source_instance_id(src),
              execution_vars(vars), card_db(db), targets(tgs), interrupted(intr), remaining_actions(rem), instruction_buffer(inst_buf), selection_var(sel_var) {}
    };

    class IActionHandler {
    public:
        virtual ~IActionHandler() = default;
        virtual void resolve(const ResolutionContext& ctx) = 0;
        virtual void resolve_with_targets([[maybe_unused]] const ResolutionContext& ctx) {}

        // Pure command generation method
        virtual void compile_action([[maybe_unused]] const ResolutionContext& ctx) {}
    };

    class EffectSystem {
    public:
        static EffectSystem& instance() {
            static EffectSystem instance;
            return instance;
        }

        void initialize();

        void register_handler(dm::core::EffectPrimitive type, std::unique_ptr<IActionHandler> handler) {
            handlers[type] = std::move(handler);
        }

        IActionHandler* get_handler(dm::core::EffectPrimitive type) {
            if (handlers.count(type)) {
                return handlers[type].get();
            }
            return nullptr;
        }

        // Migration Methods from GenericCardSystem
        void resolve_trigger(dm::core::GameState& game_state, dm::core::TriggerType trigger, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        void resolve_effect(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        void resolve_effect_with_context(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, std::map<std::string, int> execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        void resolve_effect_with_targets(dm::core::GameState& game_state, const dm::core::EffectDef& effect, const std::vector<int>& targets, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::map<std::string, int>& execution_context);
        void resolve_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, bool* interrupted = nullptr, const std::vector<dm::core::ActionDef>* remaining_actions = nullptr);

        // Generates instructions for an action without executing them immediately
        void compile_action(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::vector<dm::core::Instruction>& out_instructions);

        // Compiles a full effect into instructions
        void compile_effect(dm::core::GameState& game_state, const dm::core::EffectDef& effect, int source_instance_id, std::map<std::string, int>& execution_context, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, std::vector<dm::core::Instruction>& out_instructions);

        // Executes instructions with context synchronization
        void execute_pipeline(const ResolutionContext& ctx, const std::vector<dm::core::Instruction>& instructions);

        bool check_condition(dm::core::GameState& game_state, const dm::core::ConditionDef& condition, int source_instance_id, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, const std::map<std::string, int>& execution_context = {});
        std::vector<int> select_targets(dm::core::GameState& game_state, const dm::core::ActionDef& action, int source_instance_id, const dm::core::EffectDef& continuation, std::map<std::string, int>& execution_context);
        void delegate_selection(const ResolutionContext& ctx);

        void check_mega_last_burst(dm::core::GameState& game_state, const dm::core::CardInstance& card, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        static dm::core::PlayerID get_controller(const dm::core::GameState& game_state, int instance_id);

    private:
        EffectSystem() = default;
        EffectSystem(const EffectSystem&) = delete;
        EffectSystem& operator=(const EffectSystem&) = delete;
        std::map<dm::core::EffectPrimitive, std::unique_ptr<IActionHandler>> handlers;
        bool initialized = false;
    };
}
