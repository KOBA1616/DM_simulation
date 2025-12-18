#ifndef DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP
#define DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <map>
#include <string>
#include <variant>
#include <memory>
#include "engine/game_command/game_command.hpp"

namespace dm::engine::systems {

    using ContextValue = std::variant<int, std::string, std::vector<int>>;

    struct LoopContext {
         int index = 0;
         int max = 0;
         std::vector<int> collection;
         std::string var_name;
         bool active = false;
    };

    struct ExecutionFrame {
        std::shared_ptr<const std::vector<Instruction>> instructions;
        int pc = 0;
        LoopContext loop_ctx;
    };

    class PipelineExecutor {
    public:
        // Execution State
        std::map<std::string, ContextValue> context;
        bool execution_paused = false;
        std::string waiting_for_key;

        // Call Stack for Resume Support
        std::vector<ExecutionFrame> call_stack;

        PipelineExecutor() = default;

        // Entry point
        void execute(const std::vector<Instruction>& instructions, core::GameState& state,
                     const std::map<core::CardID, core::CardDefinition>& card_db);

        // Overload for shared_ptr
        void execute(std::shared_ptr<const std::vector<Instruction>> instructions, core::GameState& state,
                     const std::map<core::CardID, core::CardDefinition>& card_db);

        // Resume execution after input
        void resume(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db,
                    const ContextValue& input_value);

        // Context Management
        void set_context_var(const std::string& key, ContextValue value);
        ContextValue get_context_var(const std::string& key) const;
        void clear_context();

        // Helpers
        int resolve_int(const nlohmann::json& val) const;
        std::string resolve_string(const nlohmann::json& val) const;
        bool check_condition(const nlohmann::json& cond, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);

        // Instruction Handlers
        void execute_instruction(const Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_select(const Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_move(const Instruction& inst, core::GameState& state);
        void handle_modify(const Instruction& inst, core::GameState& state);
        void handle_if(const Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_loop(const Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_calc(const Instruction& inst, core::GameState& state);
        void handle_print(const Instruction& inst, core::GameState& state);

        void execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state);

        // Clone method for deep copy
        std::shared_ptr<PipelineExecutor> clone() const {
            auto copy = std::make_shared<PipelineExecutor>();
            copy->context = context;
            copy->execution_paused = execution_paused;
            copy->waiting_for_key = waiting_for_key;
            // ExecutionFrame contains shared_ptr to const instructions (safe to shallow copy)
            // LoopContext is plain struct (safe to copy)
            copy->call_stack = call_stack;
            return copy;
        }
    };

} // namespace dm::engine::systems

#endif // DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP
