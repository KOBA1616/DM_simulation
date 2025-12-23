#include "pipeline_executor.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/game_logic_system.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <set>

namespace dm::engine::systems {

    using namespace core;
    using namespace game_command;

    void PipelineExecutor::execute(const std::vector<Instruction>& instructions, GameState& state,
                                   const std::map<core::CardID, core::CardDefinition>& card_db) {
        auto shared_inst = std::make_shared<std::vector<Instruction>>(instructions);
        execute(shared_inst, state, card_db);
    }

    void PipelineExecutor::execute(std::shared_ptr<const std::vector<Instruction>> instructions, GameState& state,
                                   const std::map<core::CardID, core::CardDefinition>& card_db) {
        if ((!instructions || instructions->empty()) && call_stack.empty()) return;

        if (instructions && !instructions->empty()) {
            call_stack.push_back({instructions, 0, LoopContext{}});
        }

        execution_paused = false;

        while (!call_stack.empty() && !execution_paused) {
            auto& frame = call_stack.back();

            if (frame.pc >= (int)frame.instructions->size()) {
                call_stack.pop_back();
                continue;
            }

            const auto& inst = (*frame.instructions)[frame.pc];

            int current_stack_size = call_stack.size();

            execute_instruction(inst, state, card_db);

            if (execution_paused) break;

            if (call_stack.size() > (size_t)current_stack_size) {
                 if (inst.op != InstructionOp::IF &&
                     inst.op != InstructionOp::LOOP &&
                     inst.op != InstructionOp::REPEAT) {
                     if (current_stack_size > 0 && current_stack_size <= (int)call_stack.size()) {
                         call_stack[current_stack_size - 1].pc++;
                     }
                 }
            } else if (call_stack.size() < (size_t)current_stack_size) {
                 // Returned.
            } else {
                 if (inst.op != InstructionOp::IF &&
                     inst.op != InstructionOp::LOOP &&
                     inst.op != InstructionOp::REPEAT) {
                     frame.pc++;
                 }
            }
        }
    }

    void PipelineExecutor::resume(GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db,
                                  const ContextValue& input_value) {
        if (!waiting_for_key.empty()) {
            set_context_var(waiting_for_key, input_value);
            waiting_for_key.clear();
        }
        state.waiting_for_user_input = false;

        execute(std::shared_ptr<const std::vector<Instruction>>(), state, card_db);
    }

    void PipelineExecutor::set_context_var(const std::string& key, ContextValue value) {
        context[key] = value;
    }

    ContextValue PipelineExecutor::get_context_var(const std::string& key) const {
        auto it = context.find(key);
        if (it != context.end()) return it->second;
        return std::monostate{}; // Default
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
                } else if (type == "EXECUTE_COMMAND") {
                    GameLogicSystem::handle_execute_command(*this, state, inst);
                } else if (type == "PLAY_CARD_INTERNAL") {
                    GameLogicSystem::handle_play_card(*this, state, inst, card_db);
                }
                break;
            }
            case InstructionOp::SELECT: handle_select(inst, state, card_db); break;
            case InstructionOp::GET_STAT: handle_get_stat(inst, state, card_db); break;
            case InstructionOp::MOVE:   handle_move(inst, state); break;
            case InstructionOp::MODIFY: handle_modify(inst, state); break;
            case InstructionOp::IF:     handle_if(inst, state, card_db); break;
            case InstructionOp::LOOP:   handle_loop(inst, state, card_db); break;
            case InstructionOp::REPEAT: handle_loop(inst, state, card_db); break;
            case InstructionOp::COUNT:
            case InstructionOp::MATH:   handle_calc(inst, state); break;
            case InstructionOp::PRINT:  handle_print(inst, state); break;
            default: break;
        }
    }

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

    void PipelineExecutor::execute_command(std::unique_ptr<dm::engine::game_command::GameCommand> cmd, core::GameState& state) {
        state.execute_command(std::move(cmd));
    }

    void PipelineExecutor::handle_select(const Instruction& inst, GameState& state,
                                         const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$selection");

        if (context.count(out_key)) return;

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
                 else if (z_str == "EFFECT_BUFFER") zones.push_back(Zone::BUFFER);
            }
        }

        PlayerID player_id = state.active_player_id;

        for (PlayerID pid : {player_id, static_cast<PlayerID>(1 - player_id)}) {
            for (Zone z : zones) {
                std::vector<int> zone_indices;
                if (z == Zone::BUFFER) {
                     for(const auto& c : state.players[pid].effect_buffer) zone_indices.push_back(c.instance_id);
                } else {
                     zone_indices = state.get_zone(pid, z);
                }

                for (int instance_id : zone_indices) {
                    if (instance_id < 0) continue;
                    const auto* card_ptr = state.get_card_instance(instance_id);
                    if (!card_ptr && z == Zone::BUFFER) {
                        const auto& buf = state.players[pid].effect_buffer;
                        auto it = std::find_if(buf.begin(), buf.end(), [instance_id](const CardInstance& c){ return c.instance_id == instance_id; });
                        if (it != buf.end()) card_ptr = &(*it);
                    }

                    if (!card_ptr) continue;
                    const auto& card = *card_ptr;

                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (TargetUtils::is_valid_target(card, def, filter, state, player_id, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    } else if (card.card_id == 0) {
                         if (TargetUtils::is_valid_target(card, CardDefinition(), filter, state, player_id, pid)) {
                            valid_targets.push_back(instance_id);
                        }
                    }
                }
            }
        }

        auto count_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
        int count = resolve_int(count_val);
        if (valid_targets.empty()) {
             set_context_var(out_key, std::vector<int>{});
             return;
        }

        if (count >= (int)valid_targets.size()) {
             set_context_var(out_key, valid_targets);
             return;
        }

        execution_paused = true;
        waiting_for_key = out_key;
        state.waiting_for_user_input = true;
        state.pending_query = GameState::QueryContext{
            0, "SELECT_TARGET", {{"min", count}, {"max", count}}, valid_targets, {}
        };
    }

    void PipelineExecutor::handle_get_stat(const Instruction& inst, GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null()) return;

        std::string stat_name = resolve_string(inst.args.value("stat", ""));
        std::string out_key = inst.args.value("out", "$stat_result");

        PlayerID controller_id = state.active_player_id;
        auto v = get_context_var("$controller");
        if (std::holds_alternative<int>(v)) {
            controller_id = std::get<int>(v);
        }

        const Player& controller = state.players[controller_id];
        int result = 0;

        if (stat_name == "MANA_CIVILIZATION_COUNT") {
            std::set<std::string> civs;
            for (const auto& c : controller.mana_zone) {
                if (card_db.count(c.card_id)) {
                        const auto& cd = card_db.at(c.card_id);
                        for (const auto& civ : cd.civilizations) {
                            if (civ == Civilization::LIGHT) civs.insert("LIGHT");
                            if (civ == Civilization::WATER) civs.insert("WATER");
                            if (civ == Civilization::DARKNESS) civs.insert("DARKNESS");
                            if (civ == Civilization::FIRE) civs.insert("FIRE");
                            if (civ == Civilization::NATURE) civs.insert("NATURE");
                            if (civ == Civilization::ZERO) civs.insert("ZERO");
                        }
                }
            }
            result = (int)civs.size();
        } else if (stat_name == "SHIELD_COUNT") {
            result = (int)controller.shield_zone.size();
        } else if (stat_name == "HAND_COUNT") {
            result = (int)controller.hand.size();
        } else if (stat_name == "CARDS_DRAWN_THIS_TURN") {
            result = state.turn_stats.cards_drawn_this_turn;
        } else if (stat_name == "MANA_COUNT") {
            result = (int)controller.mana_zone.size();
        } else if (stat_name == "BATTLE_ZONE_COUNT") {
            result = (int)controller.battle_zone.size();
        } else if (stat_name == "GRAVEYARD_COUNT") {
            result = (int)controller.graveyard.size();
        }

        set_context_var(out_key, result);
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
                    auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                    virtual_count = resolve_int(c_val);
                } else if (s == "DECK_BOTTOM") {
                     is_virtual_target = true;
                     virtual_target_type = "DECK_BOTTOM";
                     auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                     virtual_count = resolve_int(c_val);
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
        else if (to_zone_str == "BUFFER") to_zone = Zone::BUFFER;
        else if (to_zone_str == "STACK") to_zone = Zone::STACK;

        bool to_bottom = false;
        if (inst.args.contains("to_bottom") && inst.args["to_bottom"].is_boolean()) {
            to_bottom = inst.args["to_bottom"];
        }

        if (is_virtual_target) {
            PlayerID pid = state.active_player_id;
            const auto& deck = state.players[pid].deck;

            if (virtual_target_type == "DECK_TOP") {
                int available = (int)deck.size();
                int count = std::min(virtual_count, available);
                for (int i = 0; i < count; ++i) {
                     if (available - 1 - i >= 0) {
                         targets.push_back(deck[available - 1 - i].instance_id);
                     }
                }
            } else if (virtual_target_type == "DECK_BOTTOM") {
                 int available = (int)deck.size();
                 int count = std::min(virtual_count, available);
                 for (int i = 0; i < count; ++i) {
                     targets.push_back(deck[i].instance_id);
                 }
            }
        }

        for (int id : targets) {
             const CardInstance* card_ptr = state.get_card_instance(id);
             if (!card_ptr) {
                 for (const auto& p : state.players) {
                     for (const auto& c : p.effect_buffer) {
                         if (c.instance_id == id) {
                             card_ptr = &c;
                             break;
                         }
                     }
                     if (card_ptr) break;
                 }
             }

             if (!card_ptr) continue;

             PlayerID owner = card_ptr->owner;
             if (owner > 1) {
                 if (state.card_owner_map.size() > (size_t)id) owner = state.card_owner_map[id];
                 else owner = state.active_player_id;
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
             if(!found) for(const auto& c : p.effect_buffer) if(c.instance_id == id) { from_zone = Zone::BUFFER; found = true; break; }

             if (!found) continue;

             int dest_idx = to_bottom ? 0 : -1;
             auto cmd = std::make_unique<TransitionCommand>(id, from_zone, to_zone, owner, dest_idx);
             execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_modify(const Instruction& inst, GameState& state) {
        if (inst.args.is_null()) return;

        std::string mod_type_str = resolve_string(inst.args.value("type", ""));

        if (mod_type_str == "SHUFFLE") {
             std::string target_zone = resolve_string(inst.args.value("target", ""));
             if (target_zone == "DECK") {
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
        auto val_json = inst.args.contains("value") ? inst.args["value"] : nlohmann::json(0);
        int val = resolve_int(val_json);
        std::string str_val = resolve_string(inst.args.value("str_value", ""));

        if (mod_type_str == "TAP") type = MutateCommand::MutationType::TAP;
        else if (mod_type_str == "UNTAP") type = MutateCommand::MutationType::UNTAP;
        else if (mod_type_str == "POWER_ADD") type = MutateCommand::MutationType::POWER_MOD;
        else if (mod_type_str == "ADD_KEYWORD") type = MutateCommand::MutationType::ADD_KEYWORD;
        else if (mod_type_str == "REMOVE_KEYWORD") type = MutateCommand::MutationType::REMOVE_KEYWORD;
        else if (mod_type_str == "ADD_PASSIVE") type = MutateCommand::MutationType::ADD_PASSIVE_EFFECT;
        else if (mod_type_str == "ADD_COST_MODIFIER") type = MutateCommand::MutationType::ADD_COST_MODIFIER;
        else if (mod_type_str == "STAT") {
            StatCommand::StatType s_type;
            std::string stat_name = resolve_string(inst.args.value("stat", ""));
            if (stat_name == "CARDS_DRAWN") s_type = StatCommand::StatType::CARDS_DRAWN;
            else if (stat_name == "CARDS_DISCARDED") s_type = StatCommand::StatType::CARDS_DISCARDED;
            else if (stat_name == "CREATURES_PLAYED") s_type = StatCommand::StatType::CREATURES_PLAYED;
            else if (stat_name == "SPELLS_CAST") s_type = StatCommand::StatType::SPELLS_CAST;
            else return;

            auto cmd = std::make_unique<StatCommand>(s_type, val);
            execute_command(std::move(cmd), state);
            return;
        }
        else return;

        // --- NEW: Intercept specific keywords to create PASSIVE EFFECTS instead ---
        if (type == MutateCommand::MutationType::ADD_KEYWORD) {
            std::vector<PassiveType> p_types;
            if (str_val == "CANNOT_ATTACK") p_types.push_back(PassiveType::CANNOT_ATTACK);
            else if (str_val == "CANNOT_BLOCK") p_types.push_back(PassiveType::CANNOT_BLOCK);
            else if (str_val == "CANNOT_ATTACK_OR_BLOCK") {
                p_types.push_back(PassiveType::CANNOT_ATTACK);
                p_types.push_back(PassiveType::CANNOT_BLOCK);
            }

            if (!p_types.empty()) {
                // For each target, apply each passive effect
                // Since MutateCommand is 1-to-1, we iterate targets here.
                for (int id : targets) {
                    for (auto pt : p_types) {
                        PassiveEffect eff;
                        eff.type = pt;
                        eff.value = 0;
                        eff.turns_remaining = inst.args.value("duration", 1);
                        eff.controller = state.active_player_id;
                        eff.specific_targets = std::vector<int>{id};

                        // We set source if available, though for ADD_KEYWORD it is often the card itself or effect source
                        int source_id = -1;
                        auto v = get_context_var("$source");
                        if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
                        eff.source_instance_id = source_id;

                        auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                        cmd->passive_effect = eff;
                        execute_command(std::move(cmd), state);
                    }
                }
                // Return early as we handled this instruction
                return;
            }
        }
        // -----------------------------------------------------------------------

        if (type == MutateCommand::MutationType::ADD_PASSIVE_EFFECT) {
             PassiveEffect eff;
             eff.target_filter = inst.args.value("filter", FilterDef{});
             if (str_val == "LOCK_SPELL") eff.type = PassiveType::CANNOT_USE_SPELLS;
             else if (str_val == "POWER") eff.type = PassiveType::POWER_MODIFIER;
             eff.value = val;
             eff.turns_remaining = inst.args.value("duration", 1);
             int source_id = -1;
             auto v = get_context_var("$source");
             if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
             eff.source_instance_id = source_id;
             eff.controller = state.active_player_id;

             auto cmd = std::make_unique<MutateCommand>(-1, type);
             cmd->passive_effect = eff;
             execute_command(std::move(cmd), state);
             return;
        }

        if (type == MutateCommand::MutationType::ADD_COST_MODIFIER) {
             CostModifier mod;
             mod.reduction_amount = val;
             mod.condition_filter = inst.args.value("filter", FilterDef{});
             mod.turns_remaining = inst.args.value("duration", 1);
             int source_id = -1;
             auto v = get_context_var("$source");
             if (std::holds_alternative<int>(v)) source_id = std::get<int>(v);
             mod.source_instance_id = source_id;
             mod.controller = state.active_player_id;

             auto cmd = std::make_unique<MutateCommand>(-1, type);
             cmd->cost_modifier = mod;
             execute_command(std::move(cmd), state);
             return;
        }

        for (int id : targets) {
            auto cmd = std::make_unique<MutateCommand>(id, type, val, str_val);
            execute_command(std::move(cmd), state);
        }
    }

    void PipelineExecutor::handle_if(const Instruction& inst, GameState& state,
                                     const std::map<core::CardID, core::CardDefinition>& card_db) {
        if (inst.args.is_null() || !inst.args.contains("cond")) {
            call_stack.back().pc++;
            return;
        }

        bool res = check_condition(inst.args["cond"], state, card_db);

        auto block = res ? std::make_shared<std::vector<Instruction>>(inst.then_block)
                         : std::make_shared<std::vector<Instruction>>(inst.else_block);

        call_stack.back().pc++;

        call_stack.push_back({block, 0, LoopContext{}});
    }

    void PipelineExecutor::handle_loop(const Instruction& inst, GameState& state,
                                       const std::map<core::CardID, core::CardDefinition>& card_db) {
        auto& frame = call_stack.back();
        auto& ctx = frame.loop_ctx;

        if (!ctx.active) {
            ctx.active = true;
            ctx.index = 0;

            if (inst.op == InstructionOp::REPEAT || (inst.args.contains("count") && !inst.args.contains("in"))) {
                auto c_val = inst.args.contains("count") ? inst.args["count"] : nlohmann::json(1);
                ctx.max = resolve_int(c_val);
                ctx.var_name = inst.args.value("var", "$i");
                ctx.collection.clear();
            } else {
                 ctx.var_name = inst.args.value("as", "$it");
                 if (inst.args.contains("in")) {
                    auto val = inst.args["in"];
                    if (val.is_string() && val.get<std::string>().rfind("$", 0) == 0) {
                        auto v = get_context_var(val.get<std::string>());
                        if (std::holds_alternative<std::vector<int>>(v)) {
                            ctx.collection = std::get<std::vector<int>>(v);
                        }
                    }
                }
                ctx.max = (int)ctx.collection.size();
            }
        }

        if (ctx.index < ctx.max) {
             if (ctx.collection.empty()) {
                 set_context_var(ctx.var_name, ctx.index);
             } else {
                 set_context_var(ctx.var_name, ctx.collection[ctx.index]);
             }

             auto block = std::make_shared<std::vector<Instruction>>(inst.then_block);

             ctx.index++;

             call_stack.push_back({block, 0, LoopContext{}});
        } else {
             ctx.active = false;
             frame.pc++;
        }
    }

    void PipelineExecutor::handle_calc(const Instruction& inst, GameState& /*state*/) {
        if (inst.args.is_null()) return;
        std::string out_key = inst.args.value("out", "$result");
        if (inst.op == InstructionOp::MATH) {
            auto l_val = inst.args.contains("lhs") ? inst.args["lhs"] : nlohmann::json(0);
            auto r_val = inst.args.contains("rhs") ? inst.args["rhs"] : nlohmann::json(0);
            int lhs = resolve_int(l_val);
            int rhs = resolve_int(r_val);
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
            int lhs = 0;
            if (cond.contains("lhs")) lhs = resolve_int(cond["lhs"]);

            int rhs = 0;
            if (cond.contains("rhs")) rhs = resolve_int(cond["rhs"]);

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
