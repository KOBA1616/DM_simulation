# Simple localization placeholder. Keep ASCII to avoid encoding issues.
TRANSLATIONS = {
    # Action Types (Comprehensive)
    "GRANT_KEYWORD": "キーワード付与",
    "MOVE_CARD": "カード移動",
    "FRIEND_BURST": "フレンド・バースト",
    "LOCK_SPELL": "呪文ロック",
    "APPLY_MODIFIER": "効果付与 (継続)",
    "PLAY_CARD": "カードをプレイ",
    "USE_ABILITY": "能力使用",
    "BREAK_SHIELD": "シールドブレイク",
    "DRAW_CARD": "カードを引く",
    "MANA_CHARGE": "マナチャージ",
    "TAP_CARD": "タップする",
    "UNTAP_CARD": "アンタップする",
    "DESTROY_CARD": "破壊する",
    "DISCARD_CARD": "手札を捨てる",
    "SEARCH_DECK": "山札から探す",
    "SHUFFLE_DECK": "山札をシャッフル",
    "LOOK_AND_ADD": "見て手札に加える",
    "LOOK_TO_BUFFER": "山札/手札を見る (バッファへ)",
    "SELECT_FROM_BUFFER": "バッファから選択",
    "PLAY_FROM_BUFFER": "バッファからプレイ",
    "MOVE_BUFFER_TO_ZONE": "バッファからゾーンへ移動",
    "ADD_SHIELD": "シールド追加",
    "SEND_SHIELD_TO_GRAVE": "シールドを墓地へ",
    "SEND_TO_DECK_BOTTOM": "山札の下に送る",
    "RETURN_TO_HAND": "手札に戻す",
    "MEKRAID": "メクレイド",
    "REVOLUTION_CHANGE": "革命チェンジ解決",
    "GET_GAME_STAT": "ゲーム統計取得",
    "COMPARE_STAT": "数値比較",
    "COUNT_CARDS": "カードを数える",
    "PASS": "パス",
    "SELECT_TARGET": "対象選択",
    "RESOLVE_EFFECT": "効果解決",
    "COST_REFERENCE": "コスト参照処理",

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
    "Untap In": "マナゾーンに置く時アンタップして置く",
    "Meta Counter": "メタカウンター",
    "Power Attacker": "パワーアタッカー",
    "Revolution Change Condition": "革命チェンジ条件設定"
}


def translate(key: str) -> str:
    """Return localized text when available, otherwise echo the key."""
    return TRANSLATIONS.get(key, key)


def tr(text: str) -> str:
    return TRANSLATIONS.get(text, text)
