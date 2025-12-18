#pragma once
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace dm::core {

    enum class InstructionOp {
        NOOP,
        // Logic
        IF,
        LOOP,       // FOREACH
        REPEAT,     // Loop N times
        // Query
        SELECT,
        // Action
        MOVE,       // Zone change
        MODIFY,     // Power/Tap/Shield
        GAME_ACTION,// High-level Game Logic (Play, Attack, etc.)
        // Calc
        COUNT,
        MATH,
        // Flow
        CALL,
        RETURN,
        WAIT_INPUT, // Interactive Pausing
        // Debug
        PRINT
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(InstructionOp, {
        {InstructionOp::NOOP, "NOOP"},
        {InstructionOp::IF, "IF"},
        {InstructionOp::LOOP, "LOOP"},
        {InstructionOp::REPEAT, "REPEAT"},
        {InstructionOp::SELECT, "SELECT"},
        {InstructionOp::MOVE, "MOVE"},
        {InstructionOp::MODIFY, "MODIFY"},
        {InstructionOp::GAME_ACTION, "GAME_ACTION"},
        {InstructionOp::COUNT, "COUNT"},
        {InstructionOp::MATH, "MATH"},
        {InstructionOp::CALL, "CALL"},
        {InstructionOp::RETURN, "RETURN"},
        {InstructionOp::WAIT_INPUT, "WAIT_INPUT"},
        {InstructionOp::PRINT, "PRINT"}
    })

    struct Instruction {
        InstructionOp op = InstructionOp::NOOP;

        // Arguments stored as JSON for flexibility
        // Example args: "filter", "count", "out", "cond", "target", "to", "value"
        nlohmann::json args;

        // Recursive blocks for control flow
        std::vector<Instruction> then_block;
        std::vector<Instruction> else_block;

        // Constructors
        Instruction() = default;
        Instruction(InstructionOp o) : op(o) {}
        Instruction(InstructionOp o, nlohmann::json a) : op(o), args(a) {}
    };

    // Forward declaration for recursion
    inline void from_json(const nlohmann::json& j, Instruction& i);

    inline void to_json(nlohmann::json& j, const Instruction& i) {
        j = i.args; // Start with args
        j["op"] = i.op;
        if (!i.then_block.empty()) j["then"] = i.then_block;
        if (!i.else_block.empty()) j["else"] = i.else_block;
    }

    inline void from_json(const nlohmann::json& j, Instruction& i) {
        if (j.contains("op")) {
            j.at("op").get_to(i.op);
        } else {
            i.op = InstructionOp::NOOP;
        }

        // Copy everything to args first
        i.args = j;

        // Remove reserved keys from args
        i.args.erase("op");
        i.args.erase("then");
        i.args.erase("else");

        // Parse blocks
        if (j.contains("then")) {
             j.at("then").get_to(i.then_block);
        }
        if (j.contains("else")) {
             j.at("else").get_to(i.else_block);
        }
    }

}
