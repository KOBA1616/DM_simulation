# Simple localization placeholder. Keep ASCII to avoid encoding issues.
TRANSLATIONS = {
    # Action Types
    "GRANT_KEYWORD": "キーワード付与",
    "MOVE_CARD": "カード移動",
    "FRIEND_BURST": "フレンド・バースト",
    "LOCK_SPELL": "呪文ロック",
    "APPLY_MODIFIER": "効果付与",

    # Zones
    "HAND": "手札",
    "BATTLE_ZONE": "バトルゾーン",
    "GRAVEYARD": "墓地",
    "MANA_ZONE": "マナゾーン",
    "SHIELD_ZONE": "シールドゾーン",
    "DECK_BOTTOM": "デッキ下",
    "DECK_TOP": "デッキ上",
    "DECK": "デッキ",

    # Civilizations
    "LIGHT": "光",
    "WATER": "水",
    "DARKNESS": "闇",
    "FIRE": "火",
    "NATURE": "自然",
    "ZERO": "無色",

    # Labels
    "Destination Zone": "移動先ゾーン",
    "Keyword": "キーワード",
    "Duration (Turns)": "持続ターン数",
    "Race (e.g. Fire Bird)": "種族 (例: ファイアー・バード)",

    # Common
    "Name": "カード名",
    "Civilization": "文明",
    "Type": "タイプ",
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

    # Filter
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
    "Ignore": "無視",
    "Yes (True)": "はい",
    "No (False)": "いいえ",
    "Selection Mode": "選択モード",
    "All/Any": "すべて/任意",
    "Fixed Number": "固定数",
    "Filter Help": "対象を選択するための条件を指定します。",
    "Include BATTLE_ZONE in target selection": "バトルゾーンを対象に含める",

    # Reaction
    "Reaction Abilities": "リアクション能力 (手札誘発)",
    "Add": "追加",
    "Remove": "削除",
    "Reaction Details": "詳細設定",
    "Trigger Event": "トリガー条件",
    "Civilization Match Required": "文明一致が必要",
    "Min Mana Required": "必要マナ枚数",
    "Zone": "発動ゾーン",
    "Cost / Requirement": "コスト/条件値",

    # Keywords
    "Speed Attacker": "スピードアタッカー",
    "Blocker": "ブロッカー",
    "Slayer": "スレイヤー",
    "Double Breaker": "W・ブレイカー",
    "Triple Breaker": "T・ブレイカー",
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
    "Revolution Change Condition": "革命チェンジ条件設定"
}


def translate(key: str) -> str:
    """Return localized text when available, otherwise echo the key."""
    return TRANSLATIONS.get(key, key)


def tr(text: str) -> str:
    return TRANSLATIONS.get(text, text)
