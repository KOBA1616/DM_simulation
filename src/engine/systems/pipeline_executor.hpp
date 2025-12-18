#ifndef DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP
#define DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/instruction.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <list>
#include <map>
#include <string>
#include <variant>
#include <memory>
#include "engine/game_command/game_command.hpp"

namespace dm::engine::systems {

    using ContextValue = std::variant<int, std::string, std::vector<int>>;

    struct LoopState {
        int current_iter;
        int max_iter;
        std::string var_name;
        std::vector<int> collection; // For-each
        bool is_foreach;
    };

    struct ExecutionFrame {
        const std::vector<dm::core::Instruction>* instructions;
        int index;
        std::optional<LoopState> loop_state;
    };

    class PipelineExecutor {
    public:
        // Execution State
        std::map<std::string, ContextValue> context;
        bool execution_paused = false;
        std::string waiting_for_key; // The context key we are waiting for input to populate

        // Call Stack for Resume Support
        std::vector<ExecutionFrame> call_stack;

        // Storage for dynamically generated instructions (to ensure pointer validity)
        std::list<std::vector<dm::core::Instruction>> dynamic_store;

        PipelineExecutor() = default;

        // Entry point
        void execute(const std::vector<dm::core::Instruction>& instructions, core::GameState& state,
                     const std::map<core::CardID, core::CardDefinition>& card_db);

        // Inject dynamic instructions (moves ownership to executor)
        void inject_instructions(std::vector<dm::core::Instruction>&& instructions);

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
        void execute_instruction(const dm::core::Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_select(const dm::core::Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_move(const dm::core::Instruction& inst, core::GameState& state);
        void handle_modify(const dm::core::Instruction& inst, core::GameState& state);
        void handle_if(const dm::core::Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_loop(const dm::core::Instruction& inst, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_calc(const dm::core::Instruction& inst, core::GameState& state);
        void handle_print(const dm::core::Instruction& inst, core::GameState& state);

        void execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state);

    private:
        void run_loop(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
    };

} // namespace dm::engine::systems

#endif // DM_ENGINE_SYSTEMS_PIPELINE_EXECUTOR_HPP
