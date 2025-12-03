#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <bitset>
#include <optional>
#include "card_json_types.hpp" // Include for FilterDef

namespace dm::core {

    // 3.2 カードデータ構造
    
    // AltCostKeywords flags and other boolean properties
    struct CardKeywords {
        bool g_zero : 1;
        bool revolution_change : 1;
        bool mach_fighter : 1;
        bool g_strike : 1;
        bool speed_attacker : 1;
        bool blocker : 1;
        bool slayer : 1;          // Added
        bool double_breaker : 1;  // Added
        bool triple_breaker : 1;  // Added
        bool power_attacker : 1;  // Added (Flag only, value in definition)
        bool shield_trigger : 1;
        bool evolution : 1;
        // Triggers
        bool cip : 1;             // Comes Into Play
        bool at_attack : 1;       // Attack Trigger
        bool at_block : 1;        // Block Trigger
        bool at_start_of_turn : 1;
        bool at_end_of_turn : 1;
        bool destruction : 1;     // Destruction Trigger

        bool hyper_energy : 1;    // Added for Phase 5 (Hyper Energy)
    };

    // Mode Selection: ModalEffectGroup 構造体による複数選択管理 [Q71]
    struct ModalEffectGroup {
        int group_id;
        std::string description;
        // Details on effects would go here or be referenced by ID
    };

    struct CardDefinition {
        CardID id;
        std::string name;
        Civilization civilization;
        CardType type;
        int cost;
        int power; // POWER_INFINITY for infinite
        int power_attacker_bonus; // Added for Power Attacker value
        std::vector<std::string> races; // Storing as string for now, could be enum later
        
        CardKeywords keywords;

        // Filter Parsing: CSVロード時に文字列条件（"OPP_TAPPED"等）をID化して保持 [Q50, Q55]
        std::vector<int> filter_ids; 

        // Modes
        std::vector<ModalEffectGroup> modes;

        // Revolution Change Condition (Data-driven)
        // If has_value(), this filter applies to the attacking creature.
        std::optional<FilterDef> revolution_change_condition;
    };

}
