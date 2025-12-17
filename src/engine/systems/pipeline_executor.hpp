#pragma once
#include "core/instruction.hpp"
#include "core/game_state.hpp"
#include "core/game_event.hpp"
#include "engine/game_command/game_command.hpp"
#include <map>
#include <variant>
#include <string>
#include <vector>

namespace dm::engine::systems {

    // Context variable types
    using ContextValue = std::variant<int, std::string, bool, std::vector<int>>;

    class PipelineExecutor {
    public:
        PipelineExecutor() = default;

        // Execute a list of instructions
        void execute(const std::vector<core::Instruction>& instructions, core::GameState& state,
                     const std::map<core::CardID, core::CardDefinition>& card_db);

        // Resume execution (Phase 7 feature stub, currently just executes synchronously)
        // void resume(core::GameState& state, int query_id, int selection_index);

        // Context management
        void set_context_var(const std::string& key, ContextValue value);
        ContextValue get_context_var(const std::string& key) const;

        // Helper to reset context (e.g. between card effects)
        void clear_context();

    private:
        std::map<std::string, ContextValue> context;
        bool execution_paused = false;

        // Execution primitives
        void execute_instruction(const core::Instruction& inst, core::GameState& state,
                                 const std::map<core::CardID, core::CardDefinition>& card_db);

        // Command execution wrapper (tracks history)
        void execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state);

        // Op handlers
        void handle_select(const core::Instruction& inst, core::GameState& state,
                           const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_move(const core::Instruction& inst, core::GameState& state);
        void handle_modify(const core::Instruction& inst, core::GameState& state);
        void handle_if(const core::Instruction& inst, core::GameState& state,
                       const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_loop(const core::Instruction& inst, core::GameState& state,
                         const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_repeat(const core::Instruction& inst, core::GameState& state,
                         const std::map<core::CardID, core::CardDefinition>& card_db);
        void handle_calc(const core::Instruction& inst, core::GameState& state); // COUNT, MATH
        void handle_print(const core::Instruction& inst, core::GameState& state);

        // Utils
        int resolve_int(const nlohmann::json& val) const;
        std::string resolve_string(const nlohmann::json& val) const;
        bool check_condition(const nlohmann::json& cond, core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
    };

}
