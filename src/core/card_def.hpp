#pragma once
#include "types.hpp"
#include <string>
#include <vector>
#include <bitset>
#include <optional>
#include "card_json_types.hpp" // Include for FilterDef
#include <algorithm> // For std::find

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
        bool slayer : 1;
        bool double_breaker : 1;
        bool triple_breaker : 1;
        bool power_attacker : 1;
        bool shield_trigger : 1;
        bool evolution : 1;
        bool neo : 1;             // Added for Step 2-4 (NEO / G-NEO)
        // Triggers
        bool cip : 1;             // Comes Into Play
        bool at_attack : 1;       // Attack Trigger
        bool at_block : 1;        // Block Trigger
        bool at_start_of_turn : 1;
        bool at_end_of_turn : 1;
        bool destruction : 1;     // Destruction Trigger

        bool just_diver : 1;      // Just Diver
        bool hyper_energy : 1;    // Phase 5 (Hyper Energy)
        bool meta_counter_play : 1; // Phase 5 (Meta Counter)
        bool shield_burn : 1;     // Shield Incineration (Burn)
        bool untap_in : 1;        // Step 1-2 (Multi-color Tap-in Exception)
    };

    // Mode Selection: ModalEffectGroup 構造体による複数選択管理 [Q71]
    struct ModalEffectGroup {
        int group_id;
        std::string description;
        // Details on effects would go here or be referenced by ID
    };

    // Generic Hand Trigger Definition
    struct HandTrigger {
        TriggerType trigger_type; // e.g. AT_END_OF_TURN
        ConditionDef condition;   // e.g. OPPONENT_PLAYED_WITHOUT_MANA
        // Action is implicitly "Play this card for 0" for now, or can be expanded later
    };

    struct CardDefinition {
        CardID id;
        std::string name;
        std::vector<Civilization> civilizations; // Changed from single civilization to vector
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

        // Hand Triggers (e.g. Meta Counter, Ninja Strike)
        std::vector<HandTrigger> hand_triggers;

        // Reaction Abilities
        std::vector<ReactionAbility> reaction_abilities;

        // Helper to check for a specific civilization
        bool has_civilization(Civilization civ) const {
            if (civ == Civilization::NONE) return civilizations.empty();
            if (civ == Civilization::ZERO) {
                 // If purely ZERO (Colorless), it should be explicitly in the list?
                 // Or implied if list is empty? Assuming explicit ZERO.
                 return std::find(civilizations.begin(), civilizations.end(), Civilization::ZERO) != civilizations.end();
            }
            return std::find(civilizations.begin(), civilizations.end(), civ) != civilizations.end();
        }
    };

}
