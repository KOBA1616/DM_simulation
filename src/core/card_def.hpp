#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <bitset>

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
        bool shield_trigger : 1;
        bool evolution : 1;
        // Triggers
        bool cip : 1;             // Comes Into Play
        bool at_attack : 1;       // Attack Trigger
        bool at_block : 1;        // Block Trigger
        bool at_start_of_turn : 1;
        bool at_end_of_turn : 1;
        bool destruction : 1;     // Destruction Trigger
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
        std::vector<std::string> races; // Storing as string for now, could be enum later
        
        CardKeywords keywords;

        // Filter Parsing: CSVロード時に文字列条件（"OPP_TAPPED"等）をID化して保持 [Q50, Q55]
        std::vector<int> filter_ids; 

        // Modes
        std::vector<ModalEffectGroup> modes;
    };

}
