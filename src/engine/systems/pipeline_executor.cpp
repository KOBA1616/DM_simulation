#include "pipeline_executor.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/game_logic_system.hpp"
#include <iostream>
#include <algorithm>
#include <random>

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
        switch (inst.op) {
            case InstructionOp::GAME_ACTION: {
                if (inst.args.is_null()) return;
                std::string type = resolve_string(inst.args.value("type", ""));
                if (type == "PLAY_CARD") {
                    GameLogicSystem::handle_play_card(*this, state, inst, card_db);
                } else if (type == "RESOLVE_PLAY") {
                    GameLogicSystem::handle_resolve_play(*this, state, inst, card_db);
                } else if (type == "ATTACK") {
                    GameLogicSystem::handle_attack(*this, state, inst, card_db);
                } else if (type == "BLOCK") {
                    GameLogicSystem::handle_block(*this, state, inst, card_db);
                } else if (type == "RESOLVE_BATTLE") {
                    GameLogicSystem::handle_resolve_battle(*this, state, inst, card_db);
                } else if (type == "BREAK_SHIELD") {
                    GameLogicSystem::handle_break_shield(*this, state, inst, card_db);
                } else if (type == "MANA_CHARGE") {
                    GameLogicSystem::handle_mana_charge(*this, state, inst);
                } else if (type == "RESOLVE_REACTION") {
                    GameLogicSystem::handle_resolve_reaction(*this, state, inst, card_db);
                } else if (type == "USE_ABILITY") {
                    GameLogicSystem::handle_use_ability(*this, state, inst, card_db);
                } else if (type == "SELECT_TARGET") {
                    GameLogicSystem::handle_select_target(*this, state, inst);
                }
                break;
            }
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
            if (s.rfind("$", 0) == 0) {
                auto v = get_context_var(s);
                if (std::holds_alternative<int>(v)) return std::get<int>(v);
            }
        }
        return 0;
    }

    std::string PipelineExecutor::resolve_string(const nlohmann::json& val) const {
        if (val.is_string()) {
            std::string s = val.get<std::string>();
            if (s.rfind("$", 0) == 0) {
                auto v = get_context_var(s);
                if (std::holds_alternative<std::string>(v)) return std::get<std::string>(v);
            }
            return s;
        }
        return "";
    }

    // --- Handlers ---

    void PipelineExecutor::execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state) {
        state.execute_command(std::move(cmd));
    }

    void PipelineExecutor::handle_select(const Instruction& inst, GameState& state,
                                         const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$selection");

        FilterDef filter = inst.args.value("filter", FilterDef{});
        std::vector<int> valid_targets;

        std::vector<Zone> zones;
        if (filter.zones.empty()) {
            zones = {Zone::BATTLE, Zone::HAND, Zone::MANA, Zone::SHIELD};
        } else {
            for (const auto& z_str : filter.zones) {
                 if (z_str == "BATTLE_ZONE") zones.push_back(Zone::BATTLE);
                 else if (z_str == "HAND") zones.push_back(Zone::HAND);
                 else if (z_str == "MANA_ZONE") zones.push_back(Zone::MANA);
                 else if (z_str == "SHIELD_ZONE") zones.push_back(Zone::SHIELD);
                 else if (z_str == "GRAVEYARD") zones.push_back(Zone::GRAVEYARD);
                 else if (z_str == "DECK") zones.push_back(Zone::DECK);
            }
        }

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
        bool is_virtual_target = false;
        std::string virtual_target_type = "";
        int virtual_count = 0;

        if (inst.args.contains("target")) {
            auto target_val = inst.args["target"];
            if (target_val.is_string()) {
                std::string s = target_val.get<std::string>();
                if (s.rfind("$", 0) == 0) {
                    auto v = get_context_var(s);
                    if (std::holds_alternative<std::vector<int>>(v)) {
                        targets = std::get<std::vector<int>>(v);
                    } else if (std::holds_alternative<int>(v)) {
                        targets.push_back(std::get<int>(v));
                    }
                } else if (s == "DECK_TOP") {
                    is_virtual_target = true;
                    virtual_target_type = "DECK_TOP";
                    virtual_count = resolve_int(inst.args.value("count", 1));
                } else if (s == "DECK_BOTTOM") {
                     is_virtual_target = true;
                     virtual_target_type = "DECK_BOTTOM";
                     virtual_count = resolve_int(inst.args.value("count", 1));
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

        if (is_virtual_target) {
            PlayerID pid = state.active_player_id;
            const auto& deck = state.players[pid].deck;

            if (virtual_target_type == "DECK_TOP") {
                int available = (int)deck.size();
                int count = std::min(virtual_count, available);
                for (int i = 0; i < count; ++i) {
                     targets.push_back(deck[available - 1 - i].instance_id);
                }
            }
        }

        for (int id : targets) {
             const CardInstance* card_ptr = state.get_card_instance(id);
             if (!card_ptr) continue;

             PlayerID owner = card_ptr->owner;
             if (owner > 1) {
                 if (state.card_owner_map.size() > (size_t)id) owner = state.card_owner_map[id];
             }

             Zone from_zone = Zone::GRAVEYARD;
             bool found = false;
             const Player& p = state.players[owner];

             for(const auto& c : p.hand) if(c.instance_id == id) { from_zone = Zone::HAND; found = true; break; }
             if(!found) for(const auto& c : p.battle_zone) if(c.instance_id == id) { from_zone = Zone::BATTLE; found = true; break; }
             if(!found) for(const auto& c : p.mana_zone) if(c.instance_id == id) { from_zone = Zone::MANA; found = true; break; }
             if(!found) for(const auto& c : p.shield_zone) if(c.instance_id == id) { from_zone = Zone::SHIELD; found = true; break; }
             if(!found) for(const auto& c : p.deck) if(c.instance_id == id) { from_zone = Zone::DECK; found = true; break; }
             if(!found) for(const auto& c : p.graveyard) if(c.instance_id == id) { from_zone = Zone::GRAVEYARD; found = true; break; }

             if (!found) continue;

             auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone, owner, -1);
             execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_modify(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::string mod_type_str = resolve_string(inst.args.value("type", ""));

        // Handle Shuffle specifically (not MutateCommand)
        if (mod_type_str == "SHUFFLE") {
             std::string target_zone = resolve_string(inst.args.value("target", ""));
             if (target_zone == "DECK") {
                 // Shuffle active player's deck
                 PlayerID pid = state.active_player_id;
                 auto& deck = state.players[pid].deck;
                 std::shuffle(deck.begin(), deck.end(), state.rng);
             }
             return;
        }

        std::vector<int> targets;
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

        MutateCommand::MutationType type;
        int val = resolve_int(inst.args.value("value", 0));
        std::string str_val = resolve_string(inst.args.value("str_value", ""));

        if (mod_type_str == "TAP") type = MutateCommand::MutationType::TAP;
        else if (mod_type_str == "UNTAP") type = MutateCommand::MutationType::UNTAP;
        else if (mod_type_str == "POWER_ADD") type = MutateCommand::MutationType::POWER_MOD;
        else if (mod_type_str == "ADD_KEYWORD") type = MutateCommand::MutationType::ADD_KEYWORD;
        else if (mod_type_str == "REMOVE_KEYWORD") type = MutateCommand::MutationType::REMOVE_KEYWORD;
        else return;

        for (int id : targets) {
            auto cmd = std::make_unique<MutateCommand>(id, type, val, str_val);
            execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_if(const Instruction& inst, GameState& state,
                                     const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null() || !inst.args.contains("cond")) return;

        if (check_condition(inst.args["cond"], state, card_db)) {
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

    bool PipelineExecutor::check_condition(const nlohmann::json& cond, GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (cond.is_null()) return false;

        if (cond.contains("type")) {
             std::string type = cond.value("type", "NONE");
             if (type != "NONE") {
                 core::ConditionDef def;
                 def.type = type;
                 if (cond.contains("value")) def.value = cond.value("value", 0);
                 if (cond.contains("str_val")) def.str_val = cond.value("str_val", "");
                 if (cond.contains("op")) def.op = cond.value("op", "==");
                 if (cond.contains("stat_key")) def.stat_key = cond.value("stat_key", "");

                 int source_id = -1;
                 auto v = get_context_var("$source");
                 if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
                 else if (std::holds_alternative<std::vector<int>>(v)) {
                     const auto& vec = std::get<std::vector<int>>(v);
                     if (!vec.empty()) source_id = vec[0];
                 }

                 std::map<std::string, int> exec_ctx;
                 for (const auto& kv : context) {
                     if (std::holds_alternative<int>(kv.second)) {
                         exec_ctx[kv.first] = std::get<int>(kv.second);
                     }
                 }

                 return dm::engine::ConditionSystem::instance().evaluate_def(state, def, source_id, card_db, exec_ctx);
             }
        }

        if (cond.contains("exists")) {
            std::string key = cond["exists"];
            auto v = get_context_var(key);
            if (std::holds_alternative<std::vector<int>>(v)) {
                return !std::get<std::vector<int>>(v).empty();
            }
            if (std::holds_alternative<int>(v)) return true;
        }

        if (cond.contains("op")) {
            int lhs = resolve_int(cond.value("lhs", 0));
            int rhs = resolve_int(cond.value("rhs", 0));
            std::string op = cond.value("op", "==");
            if (op == "==") return lhs == rhs;
            if (op == ">") return lhs > rhs;
            if (op == "<") return lhs < rhs;
            if (op == ">=") return lhs >= rhs;
            if (op == "<=") return lhs <= rhs;
            if (op == "!=") return lhs != rhs;
        }
        return false;
    }

}
