#include "pipeline_executor.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void PipelineExecutor::execute(const std::vector<Instruction>& instructions, GameState& state,
                                   const std::map<core::CardID, core::CardDefinition>& card_db) {
        for (const auto& inst : instructions) {
            if (execution_paused) break;
            execute_instruction(inst, state, card_db);
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

    void PipelineExecutor::execute_instruction(const Instruction& inst, GameState& state,
                                               const std::map<core::CardID, core::CardDefinition>& card_db) {
        // Safety check: ensure args is not null if we expect params
        // But nlohmann::json defaults to null. We handle it in handlers.
        switch (inst.op) {
            case InstructionOp::SELECT: handle_select(inst, state, card_db); break;
            case InstructionOp::MOVE:   handle_move(inst, state); break;
            case InstructionOp::MODIFY: handle_modify(inst, state); break;
            case InstructionOp::IF:     handle_if(inst, state, card_db); break;
            case InstructionOp::LOOP:   handle_loop(inst, state, card_db); break;
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

    void PipelineExecutor::handle_select(const Instruction& inst, GameState& state,
                                         const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$selection");

        // Use TargetUtils logic to identify valid targets based on filter
        FilterDef filter = inst.args.value("filter", FilterDef{});
        std::vector<int> valid_targets;

        // Iterate over specified zones to find candidates
        std::vector<Zone> zones;
        if (filter.zones.empty()) {
            // Default zones if not specified - BUT filter.zones is vector<string> in JSON but we need vector<Zone> or strings to check.
            // Wait, FilterDef::zones is vector<string>. TargetUtils handles string matching inside logic?
            // No, TargetUtils iterates what is PASSED to it usually, but here we need to fetch candidates first.
            // We need to parse string zones to Enum zones.
            zones = {Zone::BATTLE, Zone::HAND, Zone::MANA, Zone::SHIELD};
        } else {
            // Convert string zones to Zone enum
            for (const auto& z_str : filter.zones) {
                 if (z_str == "BATTLE_ZONE") zones.push_back(Zone::BATTLE);
                 else if (z_str == "HAND") zones.push_back(Zone::HAND);
                 else if (z_str == "MANA_ZONE") zones.push_back(Zone::MANA);
                 else if (z_str == "SHIELD_ZONE") zones.push_back(Zone::SHIELD);
                 else if (z_str == "GRAVEYARD") zones.push_back(Zone::GRAVEYARD);
                 else if (z_str == "DECK") zones.push_back(Zone::DECK);
            }
        }

        // Need source controller context (assumed Active Player for now, or from context)
        // In real impl, context should store $player
        PlayerID player_id = state.active_player_id;

        for (PlayerID pid : {player_id, static_cast<PlayerID>(1 - player_id)}) {
            for (Zone z : zones) {
                const auto& zone_indices = state.get_zone(pid, z);
                for (int instance_id : zone_indices) {
                    if (instance_id < 0) continue;
                    const auto* card_ptr = state.get_card_instance(instance_id);
                    if (!card_ptr) continue;
                    const auto& card = *card_ptr;

                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (TargetUtils::is_valid_target(card, def, filter, state, player_id, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    }
                }
            }
        }

        // Issue QueryCommand to ask the agent/player
        // But since PipelineExecutor runs synchronously for now, checking 'resume' logic
        // For Phase 6 Step 1, we just Issue the command.
        // Wait, QueryCommand just sends a signal. The loop needs to pause.
        // For now, let's implement the "Auto Select" logic if count is ALL, or Mock logic for single.
        // Or better: If valid_targets is empty, return empty.

        // Real Implementation:
        // auto cmd = std::make_unique<QueryCommand>("SELECT_TARGET", valid_targets);
        // cmd->execute(state);
        // execution_paused = true;
        // return;

        // Stub Implementation for Verification (since we don't have the AI loop integrated with this VM yet):
        // Just pick the first N valid targets.
        int count = inst.args.value("count", 1);
        std::vector<int> selection;
        for (int i = 0; i < count && i < (int)valid_targets.size(); ++i) {
            selection.push_back(valid_targets[i]);
        }
        set_context_var(out_key, selection);
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

        // Handle index (e.g., DECK_TOP, DECK_BOTTOM)
        // int index = inst.args.value("index", -1);
        // Special strings for index? "TOP" -> 0, "BOTTOM" -> -1

        for (int id : targets) {
             // Need owner, can look up from state
             PlayerID owner = 0; // Fallback
             if (id < (int)state.card_owner_map.size()) owner = state.card_owner_map[id];

             // Current zone lookup (inefficient, should use map if possible or GameState helper)
             Zone from_zone = Zone::GRAVEYARD; // TODO: Implement zone lookup helper in GameState
             // For now, TransitionCommand handles finding the card if we trust it,
             // BUT TransitionCommand requires from_zone.
             // Let's assume the caller knows, or iterate.
             // GameState::get_card_zone(id) is needed.
             // Since it's not exposed, we iterate.
             bool found = false;
             for (PlayerID p : {0, 1}) {
                 for (Zone z : {Zone::HAND, Zone::MANA, Zone::BATTLE, Zone::SHIELD, Zone::GRAVEYARD, Zone::DECK, Zone::STACK, Zone::HYPER_SPATIAL}) {
                     const auto& vec = state.get_zone(p, z);
                     if (std::find(vec.begin(), vec.end(), id) != vec.end()) {
                         from_zone = z;
                         owner = p;
                         found = true;
                         break;
                     }
                 }
                 if (found) break;
             }
             if (!found) continue; // Card not found

             auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone, owner, -1);
             cmd->execute(state);
        }
    }

    void PipelineExecutor::handle_modify(const Instruction& inst, GameState& state) {
        // Implementation for Modify (Power, Tap, etc.)
        if (inst.args.is_null()) return;

        std::vector<int> targets;
        // Resolve target (similar code to handle_move, should extract)
        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string() && target_val.get<std::string>().rfind("$", 0) == 0) {
                auto v = get_context_var(target_val.get<std::string>());
                if (std::holds_alternative<std::vector<int>>(v)) targets = std::get<std::vector<int>>(v);
                else if (std::holds_alternative<int>(v)) targets.push_back(std::get<int>(v));
            } else if (target_val.is_number()) {
                targets.push_back(target_val.get<int>());
            }
        }

        std::string mod_type_str = resolve_string(inst.args.value("type", ""));
        MutateCommand::MutationType type;
        int val = resolve_int(inst.args.value("value", 0));
        std::string str_val = resolve_string(inst.args.value("str_value", ""));

        if (mod_type_str == "TAP") type = MutateCommand::MutationType::TAP;
        else if (mod_type_str == "UNTAP") type = MutateCommand::MutationType::UNTAP;
        else if (mod_type_str == "POWER_ADD") type = MutateCommand::MutationType::POWER_MOD;
        else if (mod_type_str == "ADD_KEYWORD") type = MutateCommand::MutationType::ADD_KEYWORD;
        else if (mod_type_str == "REMOVE_KEYWORD") type = MutateCommand::MutationType::REMOVE_KEYWORD;
        else return; // Unknown type

        for (int id : targets) {
            auto cmd = std::make_unique<MutateCommand>(id, type, val, str_val);
            cmd->execute(state);
        }
    }

    void PipelineExecutor::handle_if(const Instruction& inst, GameState& state,
                                     const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null() || !inst.args.contains("cond")) return;

        if (check_condition(inst.args["cond"], state)) {
            execute(inst.then_block, state, card_db);
        } else {
            execute(inst.else_block, state, card_db);
        }
    }

    void PipelineExecutor::handle_loop(const Instruction& inst, GameState& state,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;

        std::string var_name = inst.args.value("as", "$it");
        std::vector<int> collection;

        // Resolve collection
        if (inst.args.contains("in")) {
            auto val = inst.args["in"];
            if (val.is_string() && val.get<std::string>().rfind("$", 0) == 0) {
                auto v = get_context_var(val.get<std::string>());
                if (std::holds_alternative<std::vector<int>>(v)) {
                    collection = std::get<std::vector<int>>(v);
                }
            }
        }

        for (int id : collection) {
            set_context_var(var_name, id);
            execute(inst.then_block, state, card_db);
        }
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
        else if (inst.op == InstructionOp::COUNT) {
             // Count items in a list variable
             if (inst.args.contains("in")) {
                auto val = inst.args["in"];
                if (val.is_string() && val.get<std::string>().rfind("$", 0) == 0) {
                    auto v = get_context_var(val.get<std::string>());
                    if (std::holds_alternative<std::vector<int>>(v)) {
                        set_context_var(out_key, (int)std::get<std::vector<int>>(v).size());
                    } else {
                        set_context_var(out_key, 0);
                    }
                }
             }
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
