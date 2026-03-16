#pragma once
#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "core/constants.hpp"

namespace dm::core {

    // Enums for JSON mapping
    enum class TriggerType {
        // --- ゾーン移動系 ---
        ON_PLAY,              // バトルゾーン参入（CIP: Come Into Play）
        ON_OTHER_ENTER,       // 自分の他クリーチャーが参入した時
        ON_OPPONENT_CREATURE_ENTER, // 相手クリーチャーが参入した時（Python側から移植）
        ON_DESTROY,           // バトルゾーン→墓地（破壊）
        ON_EXIT,              // バトルゾーンからの離脱（破壊・手札・マナ問わず）
        ON_DISCARD,           // 手札からの捨て（手札→墓地）
        // --- ターン・フェイズ系 ---
        TURN_START,           // ターン開始時
        ON_TURN_END,          // ターン終了時
        // --- アクション系 ---
        ON_ATTACK,            // 攻撃宣言時
        ON_ATTACK_FROM_HAND,  // 手札から攻撃（革命チェンジ等の起点）
        ON_BLOCK,             // ブロック宣言時
        ON_BATTLE_WIN,        // バトル勝利時
        ON_BATTLE_LOSE,       // バトル敗北時
        ON_CAST_SPELL,        // 呪文詠唱時
        ON_DRAW,              // カードを引いた時
        ON_OPPONENT_DRAW,     // 相手がカードを引いた時
        ON_TAP,               // タップした時
        ON_UNTAP,             // アンタップした時
        // --- シールド系 ---
        AT_BREAK_SHIELD,      // シールドブレイク時
        BEFORE_BREAK_SHIELD,  // シールドブレイク直前（置換起点）
        ON_SHIELD_ADD,        // シールドゾーンへの追加時
        // --- 特殊・常在型 ---
        S_TRIGGER,            // シールドトリガー（割り込み型）
        PASSIVE_CONST,        // 常在型能力（後方互換のため保持）
        NONE
    };

    enum class ReactionType {
        NONE,
        NINJA_STRIKE,
        STRIKE_BACK,
        REVOLUTION_0_TRIGGER
    };

    enum class TargetScope {
        SELF,
        PLAYER_SELF,
        PLAYER_OPPONENT,
        ALL_PLAYERS,
        TARGET_SELECT,
        RANDOM,
        ALL_FILTERED,
        NONE
    };

    enum class ModifierType {
        NONE,
        COST_MODIFIER,
        POWER_MODIFIER,
        GRANT_KEYWORD,
        SET_KEYWORD,
        FORCE_ATTACK,
        ADD_RESTRICTION
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(Civilization, {
        {Civilization::NONE, "NONE"},
        {Civilization::LIGHT, "LIGHT"},
        {Civilization::WATER, "WATER"},
        {Civilization::DARKNESS, "DARKNESS"},
        {Civilization::FIRE, "FIRE"},
        {Civilization::NATURE, "NATURE"},
        {Civilization::ZERO, "ZERO"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(CardType, {
        {CardType::CREATURE, "CREATURE"},
        {CardType::SPELL, "SPELL"},
        {CardType::EVOLUTION_CREATURE, "EVOLUTION_CREATURE"},
        {CardType::CROSS_GEAR, "CROSS_GEAR"},
        {CardType::CASTLE, "CASTLE"},
        {CardType::PSYCHIC_CREATURE, "PSYCHIC_CREATURE"},
        {CardType::GR_CREATURE, "GR_CREATURE"},
        {CardType::TAMASEED, "TAMASEED"}
    })

    enum class EffectPrimitive {
        DRAW_CARD,
        ADD_MANA,
        DESTROY,
        RETURN_TO_HAND,
        SEND_TO_MANA,
        TAP,
        UNTAP,
        MODIFY_POWER,
        BREAK_SHIELD,
        LOOK_AND_ADD,
        SUMMON_TOKEN,
        SEARCH_DECK_BOTTOM,
        MEKRAID,
        DISCARD,
        PLAY_FROM_ZONE,
        COST_REFERENCE,
        LOOK_TO_BUFFER,
        REVEAL_TO_BUFFER,
        SELECT_FROM_BUFFER,
        PLAY_FROM_BUFFER,
        MOVE_BUFFER_TO_ZONE,
        REVOLUTION_CHANGE,
        COUNT_CARDS,
        GET_GAME_STAT,
        APPLY_MODIFIER,
        REVEAL_CARDS,
        REGISTER_DELAYED_EFFECT,
        RESET_INSTANCE,
        SEARCH_DECK,
        SHUFFLE_DECK,
        ADD_SHIELD,
        SEND_SHIELD_TO_GRAVE,
        SEND_TO_DECK_BOTTOM,
        MOVE_TO_UNDER_CARD,
        SELECT_NUMBER,
        FRIEND_BURST,
        GRANT_KEYWORD,
        MOVE_CARD,
        CAST_SPELL,
        PUT_CREATURE,
        SELECT_OPTION,
        RESOLVE_BATTLE,
        IF,
        IF_ELSE,
        ELSE,
        NONE
    };

    // New Engine: Hybrid Command Types
    enum class CommandType {
        // Primitives
        TRANSITION,
        MUTATE,
        FLOW,
        QUERY,

        // Macros
        DRAW_CARD,
        DISCARD,
        DESTROY,
        BOOST_MANA, // Deck -> Mana (formerly MANA_CHARGE)
        TAP,
        UNTAP,
        POWER_MOD,
        ADD_KEYWORD,
        RETURN_TO_HAND,
        BREAK_SHIELD,
        SEARCH_DECK,
        SHIELD_TRIGGER,

        // New Primitives (Phase 2 Strict Enforcement)
        MOVE_CARD,
        ADD_MANA,
        SEND_TO_MANA,
        PLAYER_MANA_CHARGE,
        SEARCH_DECK_BOTTOM,
        ADD_SHIELD,
        SEND_TO_DECK_BOTTOM,

        // Expanded Set
        ATTACK_PLAYER,
        ATTACK_CREATURE,
        BLOCK,
        RESOLVE_BATTLE,
        RESOLVE_PLAY,
        RESOLVE_EFFECT,
        SHUFFLE_DECK,
        LOOK_AND_ADD,
        MEKRAID,
        REVEAL_CARDS,
        PLAY_FROM_ZONE,
        CAST_SPELL,
        SUMMON_TOKEN,
        SHIELD_BURN,
        SELECT_NUMBER,
        CHOICE,
        LOOK_TO_BUFFER,
        REVEAL_TO_BUFFER,
        SELECT_FROM_BUFFER,
        PLAY_FROM_BUFFER,
        MOVE_BUFFER_TO_ZONE,
        FRIEND_BURST,
        REGISTER_DELAYED_EFFECT,
        IF,
        IF_ELSE,
        ELSE,
        PASS,
        USE_ABILITY,
        MANA_CHARGE,
        SELECT_TARGET,

        // cards.json で使用される追加コマンドタイプ
        // 再発防止: cards.json に新コマンドタイプを追加したら必ずここにも追加すること
        APPLY_MODIFIER,      // 一時的な修正効果付与 (APPLY_MODIFIER in JSON)
        REVOLUTION_CHANGE,   // 革命チェンジ宣言/処理 (REVOLUTION_CHANGE in JSON)
        GRANT_KEYWORD,       // キーワード付与 (GRANT_KEYWORD in JSON)
        PUT_CREATURE,        // クリーチャーを場に出す (PUT_CREATURE in JSON)
        REPLACE_CARD_MOVE,   // カード移動の置換効果 (REPLACE_CARD_MOVE in JSON)
        ADD_RESTRICTION,     // 制約追加 (ADD_RESTRICTION in JSON)
        SELECT_OPTION,       // 選択肢提示 (SELECT_OPTION in JSON, alias of CHOICE)
        DRAW,                // カードを引く (DRAW_CARD の別名, FLOW内で使用)
        REPLACE_MOVE_CARD,   // 置換効果: カードの移動先を墓地に変更 (マグナム系)

        // 再発防止: 制限コマンドタイプ — カードエディタから出力される新形式
        //   JSON の "type" フィールドに直接これらの文字列が入る。
        //   duration フィールドは DURATION_OPTIONS 文字列 ("THIS_TURN" 等) で格納される。
        LOCK_SPELL,             // 呪文使用禁止
        SPELL_RESTRICTION,      // コスト指定呪文禁止 (target_filter.exact_cost で指定)
        CANNOT_PUT_CREATURE,    // クリーチャーを出す禁止
        CANNOT_SUMMON_CREATURE, // クリーチャー召喚禁止
        PLAYER_CANNOT_ATTACK,   // プレイヤーの攻撃禁止
        IGNORE_ABILITY,         // 指定タイプ/コストの能力を無視

            MOVE_BUFFER_REMAIN_TO_ZONE, // バッファ残余を指定ゾーンへ移動
        NONE
    };

    // Phase 4: Cost System Enums
    enum class CostType {
        MANA,
        TAP_CARD,
        SACRIFICE_CARD,
        RETURN_CARD,
        SHIELD_BURN,
        DISCARD
    };

    enum class ReductionType {
        PASSIVE,
        ACTIVE_PAYMENT
    };

    // JSON Structures
    struct FilterDef {
        std::optional<std::string> owner; // "SELF", "OPPONENT", "BOTH"
        std::vector<std::string> zones;   // "BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE"
        std::vector<std::string> types;   // "CREATURE", "SPELL"
        std::vector<Civilization> civilizations;
        std::vector<std::string> races;
        std::optional<int> min_cost;
        std::optional<int> max_cost;
        std::optional<int> exact_cost;  // For exact cost matching (can reference execution_context)
        std::optional<int> min_power;
        std::optional<int> max_power;
        std::optional<bool> is_tapped;
        std::optional<bool> is_blocker;
        std::optional<bool> is_evolution;
        std::optional<bool> is_card_designation;
        std::optional<int> count;

        std::optional<std::string> power_max_ref;
        std::optional<std::string> cost_ref;  // Reference to execution_context variable for exact_cost

        std::vector<FilterDef> and_conditions;

        // 再発防止: IF コマンドの target_filter に条件データを埋め込む場合に使用する:
        //   type に評価タイプ (例: OPPONENT_DRAW_COUNT, COMPARE_INPUT) → JSONキー "type" に対応
        //   value に閾値 → JSONキー "value" に対応
        //   op に比較演算子 (>=, <=, == 等) → JSONキー "op" に対応
        //   注意: types (vector) と type (optional単体) は別フィールドで非競合
        std::optional<std::string> type;   // 条件タイプ: OPPONENT_DRAW_COUNT, COMPARE_INPUT 等
        std::optional<int>         value;  // 条件の閾値
        std::optional<std::string> op;     // 比較演算子: >=, <=, ==, <, > 等
    };

    struct CostDef {
        CostType type;
        int amount;
        FilterDef filter;
        bool is_optional = false;
        std::string cost_id;
    };

    struct CostReductionDef {
        ReductionType type;
        CostDef unit_cost;
        int reduction_amount;
        int max_units = -1;
        int min_mana_cost = 0;
        std::string id; // unique identifier for the reduction (new)
        std::string name;
    };

    struct ConditionDef {
        std::string type;
        int value = 0;
        std::string str_val;
        std::string stat_key;
        std::string op;
        std::optional<FilterDef> filter;
        nlohmann::json extra_fields = nlohmann::json::object();  // Added: to handle unknown JSON fields
    };

    // フェーズ3: ConditionTree — AND/OR/NOT の再帰的条件木
    // 使い方:
    //   LEAF: op="LEAF", type/value/str_val/stat_key/op_str/filter で単一条件を表現（旧ConditionDefと対応）
    //   AND:  op="AND",  children に子ノードリスト（全て真なら誘発）
    //   OR:   op="OR",   children に子ノードリスト（いずれか真なら誘発）
    //   NOT:  op="NOT",  children[0] の反転
    // 再発防止: C++は再帰構造を直接保持できないため children を nlohmann::json で格納し
    //   エフェクト評価時にパースする。新しい条件タイプを追加したら effect_system.cpp も更新。
    // 再発防止: to_json/from_json は FilterDef の NLOHMANN マクロの後で定義すること
    //   (前方宣言の問題を避けるため、namespace以下の ConditionNode シリアライザセクションに記述)
    struct ConditionNode {
        std::string op = "LEAF"; // "LEAF" | "AND" | "OR" | "NOT"
        // LEAF 時のフィールド（旧 ConditionDef と同等）
        std::string type;
        int value = 0;
        std::string str_val;
        std::string stat_key;
        std::string op_str; // 比較演算子: ">=", "<=", "==", ">", "<"
        std::optional<FilterDef> filter;
        // AND/OR/NOT 時の子ノード（再帰を nlohmann::json で表現）
        nlohmann::json children = nlohmann::json::array();
    };
    // ConditionNode の to_json/from_json は NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(FilterDef,...) の後で定義する

    struct ModifierDef {
        ModifierType type = ModifierType::NONE;
        int value = 0;
        std::string str_val;
        ConditionDef condition;
        FilterDef filter;
    };

    // Deprecated legacy structure: ActionDef
    // NOTE: `ActionDef` remains defined for JSON deserialization compatibility.
    // New code should prefer `CommandDef`. The conversion is performed in
    // `json_loader.cpp::convert_legacy_action` and callers should migrate to
    // `CommandDef` when possible. This struct is retained to avoid breaking
    // existing card JSON files and to support incremental migration (T-05).
    //
    // DEPRECATION: Marked [[deprecated]] so builds will emit a warning when
    // this type is referenced from new code. The intent is to make usages
    // visible at compile-time and to encourage incremental replacement with
    // `CommandDef`. Final removal will be a separate, explicitly-scheduled
    // breaking change once tests and JSON coverage confirm parity.
    // 廃止期限: 2026-06-30 までに ActionDef 互換層を削除する計画。
    // 再発防止: 新規データは schema_version>=2 + CommandDef のみ許可し、
    // ActionDef が新規JSONへ再流入しないよう json_loader.cpp 側で境界を強制する。
    // ActionDef is legacy; prefer CommandDef. Removed C++ attribute for MSVC
    // compatibility to avoid compile-time attribute placement errors.
    struct ActionDef {
        EffectPrimitive type = EffectPrimitive::NONE;
        TargetScope scope = TargetScope::NONE;
        FilterDef filter;
        int value1 = 0;
        int value2 = 0;
        std::string str_val;
        std::string value;
        bool optional = false;
        bool up_to = false;  // Added: Allows player to choose 0 to N
        std::string target_player;
        std::string source_zone;
        std::string destination_zone;
        std::string target_choice;
        std::string input_value_key;
        std::string input_value_usage;
        std::string output_value_key;
        bool inverse_target = false;
        std::optional<ConditionDef> condition;
        std::vector<std::vector<ActionDef>> options;
        bool cast_spell_side = false;
    };

    struct CommandDef {
        CommandType type = CommandType::NONE;
        int instance_id = 0;
        int target_instance = 0;
        int owner_id = 0;
        TargetScope target_group = TargetScope::NONE;
        FilterDef target_filter;
        int amount = 0;
        std::string str_param;
        // 再発防止: str_val と duration はカードエディタが出力する新形式フィールド。
        //   ADD_KEYWORD の keyword、LOCK_SPELL 系の期間などで使用する。
        //   NLOHMANN マクロに必ず含めること。
        std::string str_val;      // キーワード識別子 (ADD_KEYWORD 等)
        std::string duration;     // 持続期間文字列 (DURATION_OPTIONS 値、例: "THIS_TURN")
        bool optional = false;
        std::string from_zone;
        std::string to_zone;
        std::string mutation_kind;
        std::optional<ConditionDef> condition;
        std::vector<CommandDef> if_true;
        std::vector<CommandDef> if_false;
        std::string input_value_key;
        std::string input_value_usage;
        std::string output_value_key;
        int slot_index = -1;
        int target_slot_index = -1;
        bool up_to = false;
        // Payment intent fields: represent explicit payment selection produced by editor/AI
        // These are optional and used to express `ACTIVE_PAYMENT` choices or other payment modes.
        std::string payment_mode;    // e.g. "ACTIVE_PAYMENT", "PAY_FROM_MANA"
        std::string reduction_id;    // id of cost_reductions entry
        int payment_units = 0;       // units selected when using ACTIVE_PAYMENT
        std::vector<std::vector<CommandDef>> options;
    };

    // TriggerDescriptor: EffectDef の誘発条件を多次元で記述する構造体
    // trigger_list に複数のTriggerTypeを指定するとOR結合で誘発する（《超次元の王家》のパンドラ・シフト等）
    // timing_mode:  "POST"(デフォルト,「～た時」) / "PRE"(「出る時」=置換効果起点)
    // multiplicity: "ONCE"(デフォルト,「時」) / "WHENEVER"(「たび」=複数誘発)
    // trigger_zones: ゾーン移動系トリガー用ゾーン指定（"BATTLE_ZONE","HAND","MANA" 等）
    struct TriggerDescriptor {
        std::vector<TriggerType> trigger_list;  // OR結合トリガーリスト（空なら trigger フィールドを使用）
        std::vector<std::string> trigger_zones; // ON_ZONE_ENTER/EXIT 用ゾーン指定
        std::string timing_mode  = "POST";      // "POST" | "PRE"
        std::string multiplicity = "ONCE";      // "ONCE" | "WHENEVER"
    };

    struct EffectDef {
        TriggerType trigger = TriggerType::NONE;
        TargetScope trigger_scope = TargetScope::NONE;
        FilterDef trigger_filter;
        ConditionDef condition;
        std::vector<ActionDef> actions;
        std::vector<CommandDef> commands;
        // フェーズ2追加: TriggerDescriptor（後方互換のため optional）
        // 再発防止: trigger_descriptor を使う場合は trigger フィールドを NONE のままにせず
        //   trigger_descriptor.trigger_list[0] と一致させると可読性が上がる
        std::optional<TriggerDescriptor> trigger_descriptor;
        // フェーズ3追加: ConditionTree（後方互換のため optional、設定時は condition より優先）
        std::optional<ConditionNode> condition_tree;
    };

    struct ReactionCondition {
        std::string trigger_event;
        bool civilization_match = false;
        int mana_count_min = 0;
        bool same_civilization_shield = false;
    };

    struct ReactionAbility {
        ReactionType type = ReactionType::NONE;
        int cost = 0;
        std::string zone;
        ReactionCondition condition;
    };

    struct CardData {
        int id;
        std::string name;
        int cost;
        std::vector<Civilization> civilizations;
        int power;
        CardType type; // Changed from std::string to CardType
        std::vector<std::string> races;
        std::vector<EffectDef> effects;
        std::vector<ModifierDef> static_abilities; // Added
        std::vector<EffectDef> metamorph_abilities;
        std::optional<FilterDef> evolution_condition;
        std::optional<FilterDef> revolution_change_condition;
        std::optional<std::map<std::string, bool>> keywords;
        std::vector<ReactionAbility> reaction_abilities;
        std::vector<CostReductionDef> cost_reductions;
        std::shared_ptr<CardData> spell_side;

        bool is_key_card = false;
        int ai_importance_score = 0;
    };

    void to_json(nlohmann::json& j, const CardData& c);
    void from_json(const nlohmann::json& j, CardData& c);

} // namespace dm::core

namespace nlohmann {
    template <typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& j, const std::optional<T>& opt) {
            if (opt == std::nullopt) {
                j = nullptr;
            } else {
                j = *opt;
            }
        }

        static void from_json(const json& j, std::optional<T>& opt) {
            if (j.is_null()) {
                opt = std::nullopt;
            } else {
                opt = j.get<T>();
            }
        }
    };
}

namespace dm::core {
    NLOHMANN_JSON_SERIALIZE_ENUM(TriggerType, {
        {TriggerType::NONE, "NONE"},
        // ゾーン移動系
        {TriggerType::ON_PLAY, "ON_PLAY"},
        {TriggerType::ON_OTHER_ENTER, "ON_OTHER_ENTER"},
        {TriggerType::ON_OPPONENT_CREATURE_ENTER, "ON_OPPONENT_CREATURE_ENTER"},
        {TriggerType::ON_DESTROY, "ON_DESTROY"},
        {TriggerType::ON_EXIT, "ON_EXIT"},
        {TriggerType::ON_DISCARD, "ON_DISCARD"},
        // ターン・フェイズ系
        {TriggerType::TURN_START, "TURN_START"},
        {TriggerType::ON_TURN_END, "ON_TURN_END"},
        // アクション系
        {TriggerType::ON_ATTACK, "ON_ATTACK"},
        {TriggerType::ON_ATTACK_FROM_HAND, "ON_ATTACK_FROM_HAND"},
        {TriggerType::ON_BLOCK, "ON_BLOCK"},
        {TriggerType::ON_BATTLE_WIN, "ON_BATTLE_WIN"},
        {TriggerType::ON_BATTLE_LOSE, "ON_BATTLE_LOSE"},
        {TriggerType::ON_CAST_SPELL, "ON_CAST_SPELL"},
        {TriggerType::ON_DRAW, "ON_DRAW"},
        {TriggerType::ON_OPPONENT_DRAW, "ON_OPPONENT_DRAW"},
        {TriggerType::ON_TAP, "ON_TAP"},
        {TriggerType::ON_UNTAP, "ON_UNTAP"},
        // シールド系
        {TriggerType::AT_BREAK_SHIELD, "AT_BREAK_SHIELD"},
        {TriggerType::BEFORE_BREAK_SHIELD, "BEFORE_BREAK_SHIELD"},
        {TriggerType::ON_SHIELD_ADD, "ON_SHIELD_ADD"},
        // 特殊・常在型
        {TriggerType::S_TRIGGER, "S_TRIGGER"},
        {TriggerType::PASSIVE_CONST, "PASSIVE_CONST"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(ReactionType, {
        {ReactionType::NONE, "NONE"},
        {ReactionType::NINJA_STRIKE, "NINJA_STRIKE"},
        {ReactionType::STRIKE_BACK, "STRIKE_BACK"},
        {ReactionType::REVOLUTION_0_TRIGGER, "REVOLUTION_0_TRIGGER"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(TargetScope, {
        {TargetScope::NONE, "NONE"},
        {TargetScope::SELF, "SELF"},
        {TargetScope::PLAYER_SELF, "PLAYER_SELF"},
        {TargetScope::PLAYER_OPPONENT, "PLAYER_OPPONENT"},
        {TargetScope::ALL_PLAYERS, "ALL_PLAYERS"},
        {TargetScope::TARGET_SELECT, "TARGET_SELECT"},
        {TargetScope::RANDOM, "RANDOM"},
        {TargetScope::ALL_FILTERED, "ALL_FILTERED"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(ModifierType, {
        {ModifierType::NONE, "NONE"},
        {ModifierType::COST_MODIFIER, "COST_MODIFIER"},
        {ModifierType::POWER_MODIFIER, "POWER_MODIFIER"},
        {ModifierType::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {ModifierType::SET_KEYWORD, "SET_KEYWORD"},
        {ModifierType::FORCE_ATTACK, "FORCE_ATTACK"},
        {ModifierType::ADD_RESTRICTION, "ADD_RESTRICTION"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(EffectPrimitive, {
        {EffectPrimitive::NONE, "NONE"},
        {EffectPrimitive::DRAW_CARD, "DRAW_CARD"},
        {EffectPrimitive::ADD_MANA, "ADD_MANA"},
        {EffectPrimitive::DESTROY, "DESTROY"},
        {EffectPrimitive::RETURN_TO_HAND, "RETURN_TO_HAND"},
        {EffectPrimitive::SEND_TO_MANA, "SEND_TO_MANA"},
        {EffectPrimitive::TAP, "TAP"},
        {EffectPrimitive::UNTAP, "UNTAP"},
        {EffectPrimitive::MODIFY_POWER, "MODIFY_POWER"},
        {EffectPrimitive::BREAK_SHIELD, "BREAK_SHIELD"},
        {EffectPrimitive::LOOK_AND_ADD, "LOOK_AND_ADD"},
        {EffectPrimitive::SUMMON_TOKEN, "SUMMON_TOKEN"},
        {EffectPrimitive::SEARCH_DECK_BOTTOM, "SEARCH_DECK_BOTTOM"},
        {EffectPrimitive::MEKRAID, "MEKRAID"},
        {EffectPrimitive::DISCARD, "DISCARD"},
        {EffectPrimitive::PLAY_FROM_ZONE, "PLAY_FROM_ZONE"},
        {EffectPrimitive::COST_REFERENCE, "COST_REFERENCE"},
        {EffectPrimitive::LOOK_TO_BUFFER, "LOOK_TO_BUFFER"},
        {EffectPrimitive::REVEAL_TO_BUFFER, "REVEAL_TO_BUFFER"},
        {EffectPrimitive::SELECT_FROM_BUFFER, "SELECT_FROM_BUFFER"},
        {EffectPrimitive::PLAY_FROM_BUFFER, "PLAY_FROM_BUFFER"},
        {EffectPrimitive::MOVE_BUFFER_TO_ZONE, "MOVE_BUFFER_TO_ZONE"},
        {EffectPrimitive::REVOLUTION_CHANGE, "REVOLUTION_CHANGE"},
        {EffectPrimitive::COUNT_CARDS, "COUNT_CARDS"},
        {EffectPrimitive::GET_GAME_STAT, "GET_GAME_STAT"},
        {EffectPrimitive::APPLY_MODIFIER, "APPLY_MODIFIER"},
        {EffectPrimitive::REVEAL_CARDS, "REVEAL_CARDS"},
        {EffectPrimitive::REGISTER_DELAYED_EFFECT, "REGISTER_DELAYED_EFFECT"},
        {EffectPrimitive::RESET_INSTANCE, "RESET_INSTANCE"},
        {EffectPrimitive::SEARCH_DECK, "SEARCH_DECK"},
        {EffectPrimitive::SHUFFLE_DECK, "SHUFFLE_DECK"},
        {EffectPrimitive::ADD_SHIELD, "ADD_SHIELD"},
        {EffectPrimitive::SEND_SHIELD_TO_GRAVE, "SEND_SHIELD_TO_GRAVE"},
        {EffectPrimitive::SEND_TO_DECK_BOTTOM, "SEND_TO_DECK_BOTTOM"},
        {EffectPrimitive::MOVE_TO_UNDER_CARD, "MOVE_TO_UNDER_CARD"},
        {EffectPrimitive::SELECT_NUMBER, "SELECT_NUMBER"},
        {EffectPrimitive::FRIEND_BURST, "FRIEND_BURST"},
        {EffectPrimitive::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {EffectPrimitive::MOVE_CARD, "MOVE_CARD"},
        {EffectPrimitive::CAST_SPELL, "CAST_SPELL"},
        {EffectPrimitive::PUT_CREATURE, "PUT_CREATURE"},
        {EffectPrimitive::SELECT_OPTION, "SELECT_OPTION"},
        {EffectPrimitive::RESOLVE_BATTLE, "RESOLVE_BATTLE"},
        {EffectPrimitive::IF, "IF"},
        {EffectPrimitive::IF_ELSE, "IF_ELSE"},
        {EffectPrimitive::ELSE, "ELSE"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(CommandType, {
        {CommandType::NONE, "NONE"},
        {CommandType::TRANSITION, "TRANSITION"},
        {CommandType::MUTATE, "MUTATE"},
        {CommandType::FLOW, "FLOW"},
        {CommandType::QUERY, "QUERY"},
        {CommandType::DRAW_CARD, "DRAW_CARD"},
        {CommandType::DISCARD, "DISCARD"},
        {CommandType::DESTROY, "DESTROY"},
        {CommandType::BOOST_MANA, "BOOST_MANA"},
        {CommandType::TAP, "TAP"},
        {CommandType::UNTAP, "UNTAP"},
        {CommandType::POWER_MOD, "POWER_MOD"},
        {CommandType::ADD_KEYWORD, "ADD_KEYWORD"},
        {CommandType::RETURN_TO_HAND, "RETURN_TO_HAND"},
        {CommandType::BREAK_SHIELD, "BREAK_SHIELD"},
        {CommandType::SEARCH_DECK, "SEARCH_DECK"},
        {CommandType::SHIELD_TRIGGER, "SHIELD_TRIGGER"},

        // New Primitives (Phase 2 Strict Enforcement)
        {CommandType::MOVE_CARD, "MOVE_CARD"},
        {CommandType::ADD_MANA, "ADD_MANA"},
        {CommandType::SEND_TO_MANA, "SEND_TO_MANA"},
        {CommandType::PLAYER_MANA_CHARGE, "PLAYER_MANA_CHARGE"},
        {CommandType::SEARCH_DECK_BOTTOM, "SEARCH_DECK_BOTTOM"},
        {CommandType::ADD_SHIELD, "ADD_SHIELD"},
        {CommandType::SEND_TO_DECK_BOTTOM, "SEND_TO_DECK_BOTTOM"},

        // Expanded Set
        {CommandType::ATTACK_PLAYER, "ATTACK_PLAYER"},
        {CommandType::ATTACK_CREATURE, "ATTACK_CREATURE"},
        {CommandType::BLOCK, "BLOCK"},
        {CommandType::RESOLVE_BATTLE, "RESOLVE_BATTLE"},
        {CommandType::RESOLVE_PLAY, "RESOLVE_PLAY"},
        {CommandType::RESOLVE_EFFECT, "RESOLVE_EFFECT"},
        {CommandType::SHUFFLE_DECK, "SHUFFLE_DECK"},
        {CommandType::LOOK_AND_ADD, "LOOK_AND_ADD"},
        {CommandType::MEKRAID, "MEKRAID"},
        {CommandType::REVEAL_CARDS, "REVEAL_CARDS"},
        {CommandType::PLAY_FROM_ZONE, "PLAY_FROM_ZONE"},
        {CommandType::CAST_SPELL, "CAST_SPELL"},
        {CommandType::SUMMON_TOKEN, "SUMMON_TOKEN"},
        {CommandType::SHIELD_BURN, "SHIELD_BURN"},
        {CommandType::SELECT_NUMBER, "SELECT_NUMBER"},
        {CommandType::CHOICE, "CHOICE"},
        {CommandType::LOOK_TO_BUFFER, "LOOK_TO_BUFFER"},
        {CommandType::REVEAL_TO_BUFFER, "REVEAL_TO_BUFFER"},
        {CommandType::SELECT_FROM_BUFFER, "SELECT_FROM_BUFFER"},
        {CommandType::PLAY_FROM_BUFFER, "PLAY_FROM_BUFFER"},
        {CommandType::MOVE_BUFFER_TO_ZONE, "MOVE_BUFFER_TO_ZONE"},
        {CommandType::FRIEND_BURST, "FRIEND_BURST"},
        {CommandType::REGISTER_DELAYED_EFFECT, "REGISTER_DELAYED_EFFECT"},
        {CommandType::IF, "IF"},
        {CommandType::IF_ELSE, "IF_ELSE"},
        {CommandType::ELSE, "ELSE"},
        {CommandType::PASS, "PASS"},
        {CommandType::USE_ABILITY, "USE_ABILITY"},
        {CommandType::MANA_CHARGE, "MANA_CHARGE"},
        {CommandType::SELECT_TARGET, "SELECT_TARGET"},

        // cards.json 追加コマンドタイプ
        // 再発防止: 新コマンドタイプを追加したら NLOHMANN と CommandType enum の両方を更新すること
        {CommandType::APPLY_MODIFIER, "APPLY_MODIFIER"},
        // 再発防止: 旧cards.json互換。COST_MODIFIER は APPLY_MODIFIER として扱う。
        {CommandType::APPLY_MODIFIER, "COST_MODIFIER"},
        // 再発防止: cards.json の REVOLUTION_CHANGE は C++ enum に必ず登録し、NONE 変換を防ぐ。
        {CommandType::REVOLUTION_CHANGE, "REVOLUTION_CHANGE"},
        {CommandType::GRANT_KEYWORD, "GRANT_KEYWORD"},
        {CommandType::PUT_CREATURE, "PUT_CREATURE"},
        {CommandType::REPLACE_CARD_MOVE, "REPLACE_CARD_MOVE"},
        {CommandType::ADD_RESTRICTION, "ADD_RESTRICTION"},
        {CommandType::SELECT_OPTION, "SELECT_OPTION"},
        {CommandType::DRAW, "DRAW"},
        {CommandType::REPLACE_MOVE_CARD, "REPLACE_MOVE_CARD"},
        // 再発防止: 制限コマンドタイプ — 追加したら command_system.cpp の switch にも追加すること
        {CommandType::LOCK_SPELL, "LOCK_SPELL"},
        {CommandType::SPELL_RESTRICTION, "SPELL_RESTRICTION"},
        {CommandType::CANNOT_PUT_CREATURE, "CANNOT_PUT_CREATURE"},
        {CommandType::CANNOT_SUMMON_CREATURE, "CANNOT_SUMMON_CREATURE"},
        {CommandType::PLAYER_CANNOT_ATTACK, "PLAYER_CANNOT_ATTACK"},
        {CommandType::IGNORE_ABILITY, "IGNORE_ABILITY"},
        {CommandType::MOVE_BUFFER_REMAIN_TO_ZONE, "MOVE_BUFFER_REMAIN_TO_ZONE"},
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(CostType, {
        {CostType::MANA, "MANA"},
        {CostType::TAP_CARD, "TAP_CARD"},
        {CostType::SACRIFICE_CARD, "SACRIFICE_CARD"},
        {CostType::RETURN_CARD, "RETURN_CARD"},
        {CostType::SHIELD_BURN, "SHIELD_BURN"},
        {CostType::DISCARD, "DISCARD"}
    })

    NLOHMANN_JSON_SERIALIZE_ENUM(ReductionType, {
        {ReductionType::PASSIVE, "PASSIVE"},
        {ReductionType::ACTIVE_PAYMENT, "ACTIVE_PAYMENT"}
    })

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(FilterDef, owner, zones, types, civilizations, races, min_cost, max_cost, exact_cost, min_power, max_power, is_tapped, is_blocker, is_evolution, is_card_designation, count, power_max_ref, cost_ref, and_conditions, type, value, op)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ConditionDef, type, value, str_val, stat_key, op, filter, extra_fields)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ModifierDef, type, value, str_val, condition, filter)
    // Custom from_json for ActionDef to accept legacy numeric enums and
    // non-standard scope strings (e.g. "SINGLE") produced by older editors.
    inline void from_json(const nlohmann::json& j, ActionDef& a) {
        // type: can be numeric (legacy) or string mapped via NLOHMANN macro
        if (j.contains("type")) {
            try {
                if (j.at("type").is_number()) {
                    a.type = static_cast<EffectPrimitive>(j.at("type").get<int>());
                } else {
                    j.at("type").get_to(a.type);
                }
            } catch (...) {
                a.type = EffectPrimitive::NONE;
            }
        }

        // scope: accept legacy string "SINGLE" as TARGET_SELECT, or numeric/string
        if (j.contains("scope")) {
            try {
                if (j.at("scope").is_string()) {
                    std::string s = j.at("scope").get<std::string>();
                    if (s == "SINGLE") a.scope = TargetScope::TARGET_SELECT;
                    else {
                        try { j.at("scope").get_to(a.scope); }
                        catch (...) { a.scope = TargetScope::NONE; }
                    }
                } else if (j.at("scope").is_number()) {
                    a.scope = static_cast<TargetScope>(j.at("scope").get<int>());
                }
            } catch (...) {
                a.scope = TargetScope::NONE;
            }
        }

        // Standard fields: guard against explicit JSON nulls before converting
        if (j.contains("filter") && !j.at("filter").is_null()) { try{ j.at("filter").get_to(a.filter); } catch(...){} }
        if (j.contains("value1") && !j.at("value1").is_null()) { try{ j.at("value1").get_to(a.value1); } catch(...){} }
        if (j.contains("value2") && !j.at("value2").is_null()) { try{ j.at("value2").get_to(a.value2); } catch(...){} }
        if (j.contains("str_val") && !j.at("str_val").is_null()) { try{ j.at("str_val").get_to(a.str_val); } catch(...){} }
        if (j.contains("value") && !j.at("value").is_null()) { try{ j.at("value").get_to(a.value); } catch(...){} }
        if (j.contains("optional") && !j.at("optional").is_null()) { try{ j.at("optional").get_to(a.optional); } catch(...){} }
        if (j.contains("up_to") && !j.at("up_to").is_null()) { try{ j.at("up_to").get_to(a.up_to); } catch(...){} }
        if (j.contains("target_player") && !j.at("target_player").is_null()) { try{ j.at("target_player").get_to(a.target_player); } catch(...){} }
        if (j.contains("source_zone") && !j.at("source_zone").is_null()) { try{ j.at("source_zone").get_to(a.source_zone); } catch(...){} }
        if (j.contains("destination_zone") && !j.at("destination_zone").is_null()) { try{ j.at("destination_zone").get_to(a.destination_zone); } catch(...){} }
        if (j.contains("target_choice") && !j.at("target_choice").is_null()) { try{ j.at("target_choice").get_to(a.target_choice); } catch(...){} }
        if (j.contains("input_value_key") && !j.at("input_value_key").is_null()) { try{ j.at("input_value_key").get_to(a.input_value_key); } catch(...){} }
        if (j.contains("input_value_usage") && !j.at("input_value_usage").is_null()) { try{ j.at("input_value_usage").get_to(a.input_value_usage); } catch(...){} }
        if (j.contains("output_value_key") && !j.at("output_value_key").is_null()) { try{ j.at("output_value_key").get_to(a.output_value_key); } catch(...){} }
        if (j.contains("inverse_target") && !j.at("inverse_target").is_null()) { try{ j.at("inverse_target").get_to(a.inverse_target); } catch(...){} }
        if (j.contains("cast_spell_side") && !j.at("cast_spell_side").is_null()) { try{ j.at("cast_spell_side").get_to(a.cast_spell_side); } catch(...){} }

        if (j.contains("condition") && !j.at("condition").is_null()) {
            try { j.at("condition").get_to(a.condition); } catch (...) {}
        }

        if (j.contains("options") && j.at("options").is_array()) {
            try { j.at("options").get_to(a.options); } catch (...) {}
        }
    }
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CommandDef, type, instance_id, target_instance, owner_id, target_group, target_filter, amount, str_param, str_val, duration, optional, from_zone, to_zone, mutation_kind, condition, if_true, if_false, input_value_key, input_value_usage, output_value_key, slot_index, target_slot_index, up_to, payment_mode, reduction_id, payment_units, options)

    // フェーズ3: ConditionNode の JSON シリアライズ
    // 再発防止: FilterDef の NLOHMANN マクロの後でないとコンパイルエラーになるため、ここに定義。
    inline void to_json(nlohmann::json& j, const ConditionNode& n) {
        j["op"] = n.op;
        if (n.op == "LEAF") {
            if (!n.type.empty())     j["type"]     = n.type;
            if (n.value != 0)        j["value"]    = n.value;
            if (!n.str_val.empty())  j["str_val"]   = n.str_val;
            if (!n.stat_key.empty()) j["stat_key"]  = n.stat_key;
            if (!n.op_str.empty())   j["op_str"]    = n.op_str;
            if (n.filter.has_value()) j["filter"]   = n.filter.value();
        } else {
            j["children"] = n.children;
        }
    }

    inline void from_json(const nlohmann::json& j, ConditionNode& n) {
        n.op = j.value("op", std::string("LEAF"));
        if (n.op == "LEAF") {
            n.type     = j.value("type",     std::string(""));
            n.value    = j.value("value",    0);
            n.str_val  = j.value("str_val",  std::string(""));
            n.stat_key = j.value("stat_key", std::string(""));
            n.op_str   = j.value("op_str",   std::string(""));
            if (j.contains("filter") && !j.at("filter").is_null()) {
                try { FilterDef f; j.at("filter").get_to(f); n.filter = f; } catch(...) {}
            }
        } else {
            n.children = j.value("children", nlohmann::json::array());
        }
    }

    // Manual to_json for EffectDef to exclude actions
    inline void to_json(nlohmann::json& j, const EffectDef& e) {
        j = nlohmann::json{
            {"trigger", e.trigger},
            {"trigger_scope", e.trigger_scope},
            {"trigger_filter", e.trigger_filter},
            {"condition", e.condition},
            {"commands", e.commands}
            // Explicitly excluding "actions" from output
        };
        // フェーズ2: TriggerDescriptor が設定されている場合のみ出力
        if (e.trigger_descriptor.has_value()) {
            const auto& td = e.trigger_descriptor.value();
            nlohmann::json tdj;
            if (!td.trigger_list.empty()) tdj["trigger_list"] = td.trigger_list;
            if (!td.trigger_zones.empty()) tdj["trigger_zones"] = td.trigger_zones;
            if (td.timing_mode != "POST")  tdj["timing_mode"]  = td.timing_mode;
            if (td.multiplicity != "ONCE") tdj["multiplicity"]  = td.multiplicity;
            if (!tdj.empty()) j["trigger_descriptor"] = tdj;
        }
        // フェーズ3: ConditionTree が設定されている場合のみ出力
        if (e.condition_tree.has_value()) {
            j["condition_tree"] = e.condition_tree.value();
        }
    }

    inline void from_json(const nlohmann::json& j, EffectDef& e) {
        if (j.contains("trigger")) j.at("trigger").get_to(e.trigger);
        if (j.contains("trigger_scope")) j.at("trigger_scope").get_to(e.trigger_scope);
        if (j.contains("trigger_filter")) j.at("trigger_filter").get_to(e.trigger_filter);
        if (j.contains("condition") && !j.at("condition").is_null()) j.at("condition").get_to(e.condition);
        if (j.contains("commands") && !j.at("commands").is_null()) {
            try { j.at("commands").get_to(e.commands); } catch(...) { e.commands = {}; }
        }
        if (j.contains("actions") && !j.at("actions").is_null()) {
            try { j.at("actions").get_to(e.actions); } catch(...) { e.actions = {}; }
        }
        // フェーズ2: TriggerDescriptor の読み込み
        if (j.contains("trigger_descriptor") && j.at("trigger_descriptor").is_object()) {
            TriggerDescriptor td;
            const auto& tdj = j.at("trigger_descriptor");
            if (tdj.contains("trigger_list")) {
                for (const auto& item : tdj.at("trigger_list")) {
                    TriggerType tt = TriggerType::NONE;
                    item.get_to(tt);
                    td.trigger_list.push_back(tt);
                }
            }
            if (tdj.contains("trigger_zones")) tdj.at("trigger_zones").get_to(td.trigger_zones);
            if (tdj.contains("timing_mode"))   tdj.at("timing_mode").get_to(td.timing_mode);
            if (tdj.contains("multiplicity"))  tdj.at("multiplicity").get_to(td.multiplicity);
            // trigger_listの最初の要素をtriggerフィールドとしてミラー（後方互換）
            if (!td.trigger_list.empty() && e.trigger == TriggerType::NONE) {
                e.trigger = td.trigger_list[0];
            }
            e.trigger_descriptor = td;
        }
        // フェーズ3: ConditionTree の読み込み
        if (j.contains("condition_tree") && j.at("condition_tree").is_object()) {
            ConditionNode cn;
            j.at("condition_tree").get_to(cn);
            e.condition_tree = cn;
        }
    }

    // NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(EffectDef, trigger, condition, actions, commands) -- REPLACED BY MANUAL IMPL ABOVE

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionCondition, trigger_event, civilization_match, mana_count_min, same_civilization_shield)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(ReactionAbility, type, cost, zone, condition)

    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CostDef, type, amount, filter, is_optional, cost_id)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(CostReductionDef, type, unit_cost, reduction_amount, max_units, min_mana_cost, id, name)

    inline void to_json(nlohmann::json& j, const CardData& c) {
        j = nlohmann::json{
            {"id", c.id},
            {"name", c.name},
            {"cost", c.cost},
            {"civilizations", c.civilizations},
            {"power", c.power},
            {"type", c.type},
            {"races", c.races},
            {"triggers", c.effects}, // Mapped to triggers in JSON
            {"static_abilities", c.static_abilities}, // Added
            {"metamorph_abilities", c.metamorph_abilities},
            {"evolution_condition", c.evolution_condition},
            {"revolution_change_condition", c.revolution_change_condition},
            {"keywords", c.keywords},
            {"reaction_abilities", c.reaction_abilities},
            {"cost_reductions", c.cost_reductions},
            {"is_key_card", c.is_key_card},
            {"ai_importance_score", c.ai_importance_score}
        };
        if (c.spell_side) {
            j["spell_side"] = *c.spell_side;
        } else {
            j["spell_side"] = nullptr;
        }
    }

    inline void from_json(const nlohmann::json& j, CardData& c) {
        c.id = j.value("id", 0);
        c.name = j.value("name", std::string(""));
        c.cost = j.value("cost", 0);
        if (j.contains("civilizations")) j.at("civilizations").get_to(c.civilizations); else c.civilizations = {};
        c.power = j.value("power", 0);
        // Use get_to for enum types (they need explicit conversion from string)
        if (j.contains("type")) {
            j.at("type").get_to(c.type);
        } else {
            c.type = CardType::CREATURE;
        }
        if (j.contains("races")) j.at("races").get_to(c.races); else c.races = {};

        // Support both "triggers" and "effects"
        if (j.contains("triggers")) {
            j.at("triggers").get_to(c.effects);
        } else if (j.contains("effects")) {
            j.at("effects").get_to(c.effects);
        } else {
            c.effects = {};
        }

        if (j.contains("static_abilities")) {
            j.at("static_abilities").get_to(c.static_abilities);
        } else {
            c.static_abilities = {};
        }

        if (j.contains("metamorph_abilities")) {
            if (!j.at("metamorph_abilities").is_null()) {
                try { j.at("metamorph_abilities").get_to(c.metamorph_abilities); }
                catch(...) { c.metamorph_abilities = {}; }
            } else {
                c.metamorph_abilities = {};
            }
        } else {
            c.metamorph_abilities = {};
        }

        if (j.contains("evolution_condition")) {
             auto& evo_cond = j.at("evolution_condition");
             // Handle empty string as no evolution condition
             if (evo_cond.is_string() && evo_cond.get<std::string>().empty()) {
                 c.evolution_condition = std::nullopt;
             } else {
                 evo_cond.get_to(c.evolution_condition);
             }
        }

        if (j.contains("revolution_change_condition")) {
             auto& rev_cond = j.at("revolution_change_condition");
             // Handle empty string as no revolution change condition
             if (rev_cond.is_string() && rev_cond.get<std::string>().empty()) {
                 c.revolution_change_condition = std::nullopt;
             } else {
                 rev_cond.get_to(c.revolution_change_condition);
             }
        }

        if (j.contains("keywords")) j.at("keywords").get_to(c.keywords);

        if (j.contains("reaction_abilities")) {
            j.at("reaction_abilities").get_to(c.reaction_abilities);
        } else {
            c.reaction_abilities = {};
        }

        if (j.contains("cost_reductions")) {
            j.at("cost_reductions").get_to(c.cost_reductions);
        } else {
            c.cost_reductions = {};
        }

        c.is_key_card = j.value("is_key_card", false);
        c.ai_importance_score = j.value("ai_importance_score", 0);

        if (j.contains("spell_side") && !j["spell_side"].is_null()) {
            c.spell_side = std::make_shared<CardData>();
            from_json(j.at("spell_side"), *c.spell_side);
        } else {
            c.spell_side = nullptr;
        }
    }
}
