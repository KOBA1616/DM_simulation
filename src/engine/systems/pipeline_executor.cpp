#include "pipeline_executor.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/utils/zone_utils.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void PipelineExecutor::execute(const std::vector<Instruction>& instructions, GameState& state) {
        for (const auto& inst : instructions) {
            if (execution_paused) break;
            execute_instruction(inst, state);
        }
    }

    void PipelineExecutor::set_context_var(const std::string& key, ContextValue value) {
        context[key] = value;
    }

    ContextValue PipelineExecutor::get_context_var(const std::string& key) const {
        auto it = context.find(key);
        if (it != context.end()) return it->second;
        return 0; // Default
    }

    void PipelineExecutor::clear_context() {
        context.clear();
    }

    void PipelineExecutor::execute_instruction(const Instruction& inst, GameState& state) {
        switch (inst.op) {
            case InstructionOp::SELECT: handle_select(inst, state); break;
            case InstructionOp::MOVE:   handle_move(inst, state); break;
            case InstructionOp::MODIFY: handle_modify(inst, state); break;
            case InstructionOp::IF:     handle_if(inst, state); break;
            case InstructionOp::LOOP:   handle_loop(inst, state); break;
            case InstructionOp::COUNT:
            case InstructionOp::MATH:   handle_calc(inst, state); break;
            case InstructionOp::PRINT:  handle_print(inst, state); break;
            default: break;
        }
    }

    // --- Utils ---

    int PipelineExecutor::resolve_int(const nlohmann::json& val) const {
        if (val.is_number()) return val.get<int>();
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.rfind("$", 0) == 0) { // Starts with $
                auto v = get_context_var(s);
                if (std::holds_alternative<int>(v)) return std::get<int>(v);
            }
        }
        return 0;
    }

    std::string PipelineExecutor::resolve_string(const nlohmann::json& val) const {
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.rfind("$", 0) == 0) { // Starts with $
                auto v = get_context_var(s);
                if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
            }
            return s;
        }
        return "";
    }

    // --- Handlers ---

    void PipelineExecutor::handle_select(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$selection");

        // Mocking: In real implementation, this would issue CMD_QUERY
        // For Phase 6 Step 2, we just mock it or assume simple AI usage where selection is pre-determined?
        // Actually, without an Agent connected, we cannot "Select".
        // However, we can implement "Auto Select All" if no agent.

        // Use TargetUtils-like logic later. For now, we keep the stub but make it slightly smarter if needed.
        std::vector<int> mock_selection = {};
        set_context_var(out_key, mock_selection);
    }

    void PipelineExecutor::handle_move(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::vector<int> targets;
        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string() && target_val.get<std::string>().rfind("$", 0) == 0) {
                auto v = get_context_var(target_val.get<std::string>());
                if (std::holds_alternative<std::vector<int>>(v)) {
                    targets = std::get<std::vector<int>>(v);
                } else if (std::holds_alternative<int>(v)) {
                    targets.push_back(std::get<int>(v));
                }
            } else if (target_val.is_number()) {
                targets.push_back(target_val.get<int>());
            }
        }

        std::string to_zone_str = resolve_string(inst.args.value("to", ""));
        Zone to_zone = Zone::GRAVEYARD;
        if (to_zone_str == "HAND") to_zone = Zone::HAND;
        else if (to_zone_str == "MANA") to_zone = Zone::MANA;
        else if (to_zone_str == "BATTLE") to_zone = Zone::BATTLE;
        else if (to_zone_str == "SHIELD") to_zone = Zone::SHIELD;
        else if (to_zone_str == "DECK") to_zone = Zone::DECK;

        for (int id : targets) {
             // Find current zone of the card
             Zone from_zone = Zone::GRAVEYARD; // Default
             PlayerID owner = 255;

             // Search all zones to find the card
             // Optimization: Use GameState::card_owner_map if available for owner,
             // but we still need the Zone.
             // We can iterate players.
             bool found = false;
             for (const auto& p : state.players) {
                 for (const auto& c : p.hand) if (c.instance_id == id) { from_zone = Zone::HAND; owner = p.id; found = true; break; }
                 if (found) break;
                 for (const auto& c : p.battle_zone) if (c.instance_id == id) { from_zone = Zone::BATTLE; owner = p.id; found = true; break; }
                 if (found) break;
                 for (const auto& c : p.mana_zone) if (c.instance_id == id) { from_zone = Zone::MANA; owner = p.id; found = true; break; }
                 if (found) break;
                 for (const auto& c : p.shield_zone) if (c.instance_id == id) { from_zone = Zone::SHIELD; owner = p.id; found = true; break; }
                 if (found) break;
                 for (const auto& c : p.graveyard) if (c.instance_id == id) { from_zone = Zone::GRAVEYARD; owner = p.id; found = true; break; }
                 if (found) break;
                 for (const auto& c : p.deck) if (c.instance_id == id) { from_zone = Zone::DECK; owner = p.id; found = true; break; }
                 if (found) break;
             }

             if (found) {
                 auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone, owner, 0);
                 cmd->execute(state);
             }
        }
    }

    void PipelineExecutor::handle_modify(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::vector<int> targets;
        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string() && target_val.get<std::string>().rfind("$", 0) == 0) {
                auto v = get_context_var(target_val.get<std::string>());
                if (std::holds_alternative<std::vector<int>>(v)) {
                    targets = std::get<std::vector<int>>(v);
                } else if (std::holds_alternative<int>(v)) {
                    targets.push_back(std::get<int>(v));
                }
            } else if (target_val.is_number()) {
                targets.push_back(target_val.get<int>());
            }
        }

        std::string mod_type = resolve_string(inst.args.value("type", "")); // TAP, UNTAP, POWER
        int value = resolve_int(inst.args.value("value", 0));

        MutateCommand::MutationType m_type;
        bool valid = false;

        if (mod_type == "TAP") { m_type = MutateCommand::MutationType::TAP; valid = true; }
        else if (mod_type == "UNTAP") { m_type = MutateCommand::MutationType::UNTAP; valid = true; }
        else if (mod_type == "POWER") { m_type = MutateCommand::MutationType::POWER_MOD; valid = true; }

        if (valid) {
            for (int id : targets) {
                auto cmd = std::make_unique<MutateCommand>(id, m_type, value);
                cmd->execute(state);
            }
        }
    }

    void PipelineExecutor::handle_if(const Instruction& inst, GameState& state) {
        if (inst.args.is_null() || !inst.args.contains("cond")) return;

        if (check_condition(inst.args["cond"], state)) {
            execute(inst.then_block, state);
        } else {
            execute(inst.else_block, state);
        }
    }

    void PipelineExecutor::handle_loop(const Instruction& /*inst*/, GameState& /*state*/) {
        // Implementation for Foreach
    }

    void PipelineExecutor::handle_calc(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$result");
        if (inst.op == InstructionOp::MATH) {
            int lhs = resolve_int(inst.args.value("lhs", 0));
            int rhs = resolve_int(inst.args.value("rhs", 0));
            std::string op = inst.args.value("op", "+");
            int res = 0;
            if (op == "+") res = lhs + rhs;
            else if (op == "-") res = lhs - rhs;
            else if (op == "*") res = lhs * rhs;
            else if (op == "/") res = (rhs != 0) ? lhs / rhs : 0;

            set_context_var(out_key, res);
        }
    }

    void PipelineExecutor::handle_print(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::cout << "[Pipeline] " << resolve_string(inst.args.value("msg", "")) << std::endl;
    }

    bool PipelineExecutor::check_condition(const nlohmann::json& cond, GameState& /*state*/) {
        if (cond.is_null()) return false;

        // Simple "exists": "$var" check
        if (cond.contains("exists")) {
            std::string key = cond["exists"];
            auto v = get_context_var(key);
            if (std::holds_alternative<std::vector<int>>(v)) {
                return !std::get<std::vector<int>>(v).empty();
            }
            if (std::holds_alternative<int>(v)) return true;
        }
        // Simple comparison "lhs", "op", "rhs"
        if (cond.contains("op")) {
            int lhs = resolve_int(cond.value("lhs", 0));
            int rhs = resolve_int(cond.value("rhs", 0));
            std::string op = cond.value("op", "==");
            if (op == "==") return lhs == rhs;
            if (op == ">") return lhs > rhs;
            if (op == "<") return lhs < rhs;
        }
        return false;
    }

}
