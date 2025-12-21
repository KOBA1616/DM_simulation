# -*- coding: utf-8 -*-
# Localized Japanese text.
try:
    import dm_ai_module as m
except ImportError:
    m = None

# Base translations
TRANSLATIONS = {
    # Keys for Enums will be dynamically added below if m is available
    "Destination Zone": "移動先ゾーン",
    "Keyword": "キーワード",
    "Duration (Turns)": "持続ターン数",
    "Race (e.g. Fire Bird)": "種族 (例: ファイアー・バード)",
    "Action Type": "アクションタイプ",
    "Scope": "対象スコープ",
    "String Value": "文字列値 (String Value)",
    "Mode": "モード",
    "Ref Mode": "参照モード",
    "Value 1": "値 1",
    "Value 2": "値 2",
    "Arbitrary Amount (Up to N)": "任意数 (N枚まで)",
    "Filter": "フィルタ",
    "Name": "カード名",
    "Civilization": "文明",
    "Type": "タイプ",
    "ELEMENT": "エレメント",
    "CARD": "カード",
    "Cost": "コスト",
    "Power": "パワー",
    "Races": "種族",
    "Keywords": "キーワード能力",
    "Twinpact": "ツインパクト",
    "Is Twinpact?": "ツインパクトにする",
    "--- Twinpact Spell Side ---": "--- ツインパクト呪文側 ---",
    "Spell Side Name": "呪文側の名前",
    "Effects for Spell side are managed in the tree.": "呪文側の効果は左のツリーで管理してください。",
    "AI Configuration": "AI設定",
    "Is Key Card / Combo Piece": "キーカード / コンボパーツ",
    "AI Importance Score": "AI優先度スコア",
    "Action": "アクション",
    "Basic Filter": "基本条件",
    "Stats Filter": "ステータス条件",
    "Flags Filter": "状態条件",
    "Selection": "選択数設定",
    "Zones:": "ゾーン:",
    "Types:": "カードタイプ:",
    "Civilizations:": "文明:",
    "Races:": "種族:",
    "Comma separated races (e.g. Dragon, Cyber Lord)": "カンマ区切り (例: ドラゴン, サイバーロード)",
    "Min:": "最小:",
    "Max:": "最大:",
    "Any": "指定なし",
    "Is Tapped?": "タップ状態?",
    "Is Blocker?": "ブロッカー?",
    "Is Evolution?": "進化?",
    "Card Designation": "カード指定(非エレメント)",
    "Ignore": "無視",
    "Yes (True)": "はい",
    "No (False)": "いいえ",
    "Selection Mode": "選択モード",
    "All/Any": "すべて/任意",
    "Selection_All": "すべて",
    "Selection_Any": "任意",
    "Fixed Number": "固定数",
    "Filter Help": "対象を選択するための条件を指定します。",
    "Include BATTLE_ZONE in target selection": "バトルゾーンを対象に含める",
    "Play without paying cost": "コストを支払わずにプレイする",
    "Reaction Abilities": "リアクション能力 (手札誘発)",
    "Add": "追加",
    "Remove": "削除",
    "Reaction Details": "詳細設定",
    "Trigger Event": "トリガー条件",
    "Civilization Match Required": "文明一致が必要",
    "Min Mana Required": "必要マナ枚数",
    "Zone": "発動ゾーン",
    "Cost / Requirement": "コスト/条件値",
    "ON_BLOCK_OR_ATTACK": "ブロック時または攻撃時",
    "Speed Attacker": "スピードアタッカー",
    "Blocker": "ブロッカー",
    "Slayer": "スレイヤー",
    "Double Breaker": "W・ブレイカー",
    "Triple Breaker": "T・ブレイカー",
    "World Breaker": "ワールド・ブレイカー",
    "Shield Trigger": "S・トリガー",
    "Evolution": "進化",
    "Just Diver": "ジャストダイバー",
    "Mach Fighter": "マッハファイター",
    "G Strike": "G・ストライク",
    "Hyper Energy": "ハイパーエナジー",
    "Shield Burn": "シールド焼却",
    "Revolution Change": "革命チェンジ",
    "Untap In": "アンタップイン",
    "Meta Counter": "メタカウンター",
    "Power Attacker": "パワーアタッカー",
    "Revolution Change Condition": "革命チェンジ条件設定",
    "OPPONENT_DRAW_COUNT": "相手ドロー枚数",
    "Add Reaction Ability": "リアクション能力を追加",
    "Reaction Ability": "リアクション能力",
    "W-Breaker": "W・ブレイカー",
    "T-Breaker": "T・ブレイカー",
    "G-Zero": "G・ゼロ",
    "CIP": "出た時 (CIP)",
    "At Attack": "攻撃時",
    "At Block": "ブロック時",
    "Start of Turn": "ターン開始時",
    "End of Turn": "ターン終了時",
    "On Destroy": "破壊された時",
    "Spell Side": "呪文側",
    "Edit Spell Side Properties": "呪文側のプロパティ編集",
    "Add Revolution Change": "革命チェンジを追加",
    "Special Abilities": "特殊能力",
    "Enable to generate a Spell Side node in the logic tree.": "ロジックツリーに呪文側のノードを生成します。",
    "SPELL_SIDE": "呪文側",
    "Reduce Cost by": "コスト軽減:",
    "Mode Selection": "モード選択",
    "Reference": "参照:",
    "Option": "選択肢",
    "Effect": "効果",
    "Reaction": "リアクション",
    "Game Info & Controls": "ゲーム情報 & 操作",
    "Game Status & Operations": "ゲームステータス & 操作",
    "Turn": "ターン",
    "Phase": "フェーズ",
    "Active": "アクティブ",
    "Start Sim": "シミュ開始",
    "Stop Sim": "シミュ停止",
    "Step": "ステップ",
    "Reset": "リセット",
    "AI & Tools": "AI & ツール",
    "Player Mode": "プレイヤーモード",
    "P0 (Self): Human": "P0 (自分): 人間",
    "P0 (Self): AI": "P0 (自分): AI",
    "P1 (Opp): Human": "P1 (相手): 人間",
    "P1 (Opp): AI": "P1 (相手): AI",
    "Deck Builder": "デッキビルダー",
    "Card Editor": "カードエディタ",
    "Scenario Editor": "シナリオエディタ",
    "Batch Simulation": "一括シミュレーション",
    "Deck Management": "デッキ管理",
    "Load Deck P0": "デッキ読込 P0",
    "Load Deck P1": "デッキ読込 P1",
    "God View": "ゴッドビュー (全公開)",
    "Help / Manual": "ヘルプ / マニュアル",
    "Logs": "ログ",
    "MCTS Analysis": "MCTS分析",
    "Invalid Deck": "無効なデッキ",
    "Deck must have 40 cards.": "デッキは40枚である必要があります。",
    "Failed to load deck": "デッキの読み込みに失敗しました",
    "Loaded Deck for P0": "P0のデッキを読み込みました",
    "Loaded Deck for P1": "P1のデッキを読み込みました",
    "Game Reset": "ゲームリセット",
    "No actions for card": "このカードで実行できるアクションはありません",
    "Multiple actions found. Executing first.": "複数のアクションが見つかりました。最初のアクションを実行します。",
    "Game Over! Result": "ゲームオーバー! 結果",
    "Auto-Pass": "自動パス",
    "AI Action": "AIアクション",
    "Simulation stopped.": "シミュレーション停止",
    "Error: MCTS returned None": "エラー: MCTSが結果を返しませんでした",
    "P1 Action": "P1 アクション",
    "P0 Action": "P0 アクション",
    "Search...": "検索...",
    "New Card": "新規カード",
    "Preview": "プレビュー",
    "Deck": "デッキ",
    "Save Deck": "デッキ保存",
    "Load Deck": "デッキ読込",
    "Deck must have exactly 40 cards.": "デッキはちょうど40枚である必要があります。",
    "Deck saved!": "デッキを保存しました!",
    "Unknown": "不明",
    "Card Name": "カード名",
    "Unknown Card": "不明なカード",
    "Card effects will appear here...": "カード効果がここに表示されます...",
    "Target Group": "対象グループ",
    "Command Type": "コマンドタイプ",
    "Mutation Kind": "変異種別 (Mutation Kind)",
    "String Param": "文字列パラメータ",
    "Amount": "量/数値",
    "From Zone": "移動元ゾーン",
    "To Zone": "移動先ゾーン",
    "Optional (Arbitrary Amount)": "任意 (任意数)",
    "Query Mode": "クエリモード",
    "TRANSITION": "移動 (TRANSITION)",
    "MUTATE": "変異 (MUTATE)",
    "FLOW": "フロー制御",
    "QUERY": "クエリ (QUERY)",
    "POWER_MOD": "パワー修正",
    "ADD_KEYWORD": "キーワード付与",
    "MANA_CHARGE": "マナチャージ",
}

# Add Enum mappings if module is available
if m:
    # EffectActionType
    TRANSLATIONS.update({
        m.EffectActionType.GRANT_KEYWORD: "キーワード付与",
        m.EffectActionType.MOVE_CARD: "カード移動",
        m.EffectActionType.FRIEND_BURST: "フレンド・バースト",
        m.EffectActionType.APPLY_MODIFIER: "効果付与",
        m.EffectActionType.DRAW_CARD: "カードを引く",
        m.EffectActionType.ADD_MANA: "マナ追加",
        m.EffectActionType.DESTROY: "破壊",
        m.EffectActionType.RETURN_TO_HAND: "手札に戻す",
        m.EffectActionType.SEND_TO_MANA: "マナ送りにする",
        m.EffectActionType.TAP: "タップする",
        m.EffectActionType.UNTAP: "アンタップする",
        m.EffectActionType.MODIFY_POWER: "パワー修正",
        m.EffectActionType.BREAK_SHIELD: "シールドブレイク",
        m.EffectActionType.LOOK_AND_ADD: "見て加える(サーチ)",
        m.EffectActionType.SEARCH_DECK_BOTTOM: "デッキ下サーチ",
        m.EffectActionType.MEKRAID: "メクレイド",
        m.EffectActionType.REVOLUTION_CHANGE: "革命チェンジ",
        m.EffectActionType.COUNT_CARDS: "カードカウント",
        m.EffectActionType.GET_GAME_STAT: "ゲーム統計取得",
        m.EffectActionType.REVEAL_CARDS: "カード公開",
        m.EffectActionType.RESET_INSTANCE: "カード状態リセット",
        m.EffectActionType.REGISTER_DELAYED_EFFECT: "遅延効果登録",
        m.EffectActionType.SEARCH_DECK: "デッキ探索",
        m.EffectActionType.SHUFFLE_DECK: "デッキシャッフル",
        m.EffectActionType.ADD_SHIELD: "シールド追加",
        m.EffectActionType.SEND_SHIELD_TO_GRAVE: "シールド焼却",
        m.EffectActionType.SEND_TO_DECK_BOTTOM: "デッキ下に送る",
        m.EffectActionType.MOVE_TO_UNDER_CARD: "カードの下に重ねる",
        m.EffectActionType.CAST_SPELL: "呪文を唱える",
        m.EffectActionType.PUT_CREATURE: "クリーチャーを出す",
        m.EffectActionType.COST_REFERENCE: "コスト参照/軽減",
        m.EffectActionType.SELECT_NUMBER: "数字を選択",
        m.EffectActionType.SUMMON_TOKEN: "トークン生成",
        m.EffectActionType.DISCARD: "手札を捨てる",
        m.EffectActionType.PLAY_FROM_ZONE: "ゾーンからプレイ",
        m.EffectActionType.LOOK_TO_BUFFER: "バッファへ移動(Look)",
        m.EffectActionType.SELECT_FROM_BUFFER: "バッファから選択",
        m.EffectActionType.PLAY_FROM_BUFFER: "バッファからプレイ",
        m.EffectActionType.MOVE_BUFFER_TO_ZONE: "バッファから移動",
        m.EffectActionType.SELECT_OPTION: "選択肢",
        m.EffectActionType.RESOLVE_BATTLE: "バトル解決",
    })

    # ActionType
    TRANSLATIONS.update({
        m.ActionType.PLAY_CARD: "カードをプレイ",
        m.ActionType.ATTACK_CREATURE: "クリーチャー攻撃",
        m.ActionType.ATTACK_PLAYER: "プレイヤー攻撃",
        m.ActionType.BLOCK: "ブロック",
        m.ActionType.USE_SHIELD_TRIGGER: "S・トリガー使用",
        m.ActionType.RESOLVE_EFFECT: "効果解決",
        m.ActionType.SELECT_TARGET: "対象選択",
        m.ActionType.USE_ABILITY: "能力使用",
        m.ActionType.DECLARE_REACTION: "リアクション宣言",
        m.ActionType.MANA_CHARGE: "マナチャージ",
        m.ActionType.PASS: "パス",
    })

    # TriggerType
    TRANSLATIONS.update({
        m.TriggerType.ON_PLAY: "出た時 (CIP)",
        m.TriggerType.ON_ATTACK: "攻撃する時",
        m.TriggerType.ON_DESTROY: "破壊された時",
        m.TriggerType.S_TRIGGER: "S・トリガー",
        m.TriggerType.TURN_START: "ターン開始時",
        m.TriggerType.PASSIVE_CONST: "常在効果(パッシブ)",
    })

    # Civilization
    TRANSLATIONS.update({
        m.Civilization.FIRE: "火",
        m.Civilization.WATER: "水",
        m.Civilization.NATURE: "自然",
        m.Civilization.LIGHT: "光",
        m.Civilization.DARKNESS: "闇",
        m.Civilization.ZERO: "無色",
    })

    # Zone
    TRANSLATIONS.update({
        m.Zone.HAND: "手札",
        m.Zone.BATTLE: "バトルゾーン",
        m.Zone.GRAVEYARD: "墓地",
        m.Zone.MANA: "マナゾーン",
        m.Zone.SHIELD: "シールドゾーン",
        m.Zone.DECK: "デッキ",
        m.Zone.BUFFER: "効果バッファ",
    })

    # TargetScope
    TRANSLATIONS.update({
        m.TargetScope.SELF: "自分",
        m.TargetScope.TARGET_SELECT: "対象選択",
        m.TargetScope.NONE: "なし",
    })

    # String Values that were keys (kept for compatibility or mapped to Enum str?)
    # ...

    # Also keep string keys for Enums for backward compatibility or serialization
    for enum_cls in [m.ActionType, m.EffectActionType, m.TriggerType, m.Civilization, m.Zone, m.TargetScope]:
        for member in enum_cls.__members__.values():
            if member in TRANSLATIONS:
                TRANSLATIONS[member.name] = TRANSLATIONS[member]

def translate(key) -> str:
    """Return localized text when available, otherwise echo the key."""
    # Try direct lookup (works for Enums and strings)
    res = TRANSLATIONS.get(key)
    if res is not None:
        return res

    # If key is an Enum, try looking up its name (fallback)
    if hasattr(key, "name"):
         res = TRANSLATIONS.get(key.name)
         if res is not None:
             return res

    # If key is a string and not found, return as is
    return str(key)

def tr(text: str) -> str:
    return translate(text)

def get_card_civilizations(card_data) -> list:
    """
    Returns a list of civilization names (e.g. ["FIRE", "NATURE"]) from card data.
    Handles C++ pybind11 objects and legacy dicts.
    """
    if not card_data:
        return ["COLORLESS"]

    if hasattr(card_data, 'civilizations') and card_data.civilizations:
        civs = []
        for c in card_data.civilizations:
            if hasattr(c, 'name'):
                civs.append(c.name)
            else:
                civs.append(str(c).split('.')[-1])
        return civs

    elif hasattr(card_data, 'civilization'):
        # Legacy singular
        c = card_data.civilization
        if hasattr(c, 'name'):
            return [c.name]
        return [str(c).split('.')[-1]]

    return ["COLORLESS"]

def get_card_civilization(card_data) -> str:
    """
    Returns the primary civilization name as a string.
    If multiple, returns the first one.
    """
    civs = get_card_civilizations(card_data)
    if civs:
        return civs[0]
    return "COLORLESS"
