# python/gui/localization.py

TRANSLATIONS = {
    # Window & Menus
    "Card Editor": "カードエディタ",
    "Load from JSON": "JSONから読み込み",
    "Save to JSON": "JSONへ保存",
    "File": "ファイル",
    "Open": "開く",
    "Save": "保存",
    "Close": "閉じる",

    # Card Details
    "Card Details": "カード詳細",
    "ID": "ID",
    "Name": "名前",
    "Cost": "コスト",
    "Civilization": "文明",
    "Power": "パワー",
    "Type": "タイプ",
    "Races": "種族",

    # Lists & Groups
    "Keywords": "キーワード能力",
    "Effects": "効果リスト",
    "Add Effect": "効果を追加",
    "Remove Effect": "効果を削除",
    "Reaction Abilities": "リアクション能力", # Ninja Strike etc

    # Effect Definition
    "Effect": "効果",
    "Trigger": "トリガー",
    "Condition": "発動条件",
    "Actions": "アクションリスト",
    "Add Action": "アクションを追加",
    "Remove Action": "アクションを削除",

    # Action Definition
    "Action": "アクション",
    "Action Type": "アクションタイプ",
    "Scope": "対象範囲 (Scope)",
    "Target Player": "対象プレイヤー",
    "Source Zone": "移動元ゾーン",
    "Destination Zone": "移動先ゾーン",
    "Value 1": "値 1 (数)",
    "Value 2": "値 2",
    "String Value": "文字列値",
    "Input Key": "入力変数キー",
    "Output Key": "出力変数キー",
    "Optional": "任意効果",
    "Update Action": "アクション更新",

    # Filter Definition
    "Filter": "フィルタ",
    "Owner": "所有者",
    "Zones": "ゾーン",
    "Card Types": "カードタイプ",
    "Civilizations": "文明フィルタ",
    "Races": "種族フィルタ",
    "Min Cost": "最小コスト",
    "Max Cost": "最大コスト",
    "Min Power": "最小パワー",
    "Max Power": "最大パワー",
    "Tapped": "タップ状態",
    "Blocker": "ブロッカー",
    "Evolution": "進化",
    "Count": "枚数",
    "Ignore": "指定なし",
    "True": "はい",
    "False": "いいえ",

    # Common Values
    "NONE": "なし",
    "CREATURE": "クリーチャー",
    "SPELL": "呪文",
    "EVOLUTION_CREATURE": "進化クリーチャー",
    "LIGHT": "光",
    "WATER": "水",
    "DARKNESS": "闇",
    "FIRE": "火",
    "NATURE": "自然",
    "ZERO": "ゼロ",

    # Enums - Triggers
    "ON_PLAY": "登場時 (ON_PLAY)",
    "ON_ATTACK": "攻撃時 (ON_ATTACK)",
    "ON_DESTROY": "破壊時 (ON_DESTROY)",
    "S_TRIGGER": "S・トリガー",
    "TURN_START": "ターン開始時",
    "PASSIVE_CONST": "常在効果",
    "ON_OTHER_ENTER": "他クリーチャー登場時",
    "ON_ATTACK_FROM_HAND": "手札から攻撃時 (革命チェンジ等)",

    # Enums - Actions
    "DRAW_CARD": "ドロー",
    "ADD_MANA": "マナ加速",
    "DESTROY": "破壊",
    "RETURN_TO_HAND": "手札に戻す (バウンス)",
    "SEND_TO_MANA": "マナ送りにする",
    "TAP": "タップする",
    "UNTAP": "アンタップする",
    "MODIFY_POWER": "パワー修正",
    "BREAK_SHIELD": "シールドブレイク",
    "LOOK_AND_ADD": "見て手札に加える",
    "SUMMON_TOKEN": "トークン召喚",
    "SEARCH_DECK_BOTTOM": "山札下サーチ",
    "MEKRAID": "メクレイド",
    "DISCARD": "手札を捨てる",
    "PLAY_FROM_ZONE": "ゾーンからプレイ",
    "COST_REFERENCE": "コスト参照/軽減",
    "LOOK_TO_BUFFER": "バッファへ見る",
    "SELECT_FROM_BUFFER": "バッファから選択",
    "PLAY_FROM_BUFFER": "バッファからプレイ",
    "MOVE_BUFFER_TO_ZONE": "バッファから移動",
    "REVOLUTION_CHANGE": "革命チェンジ",
    "COUNT_CARDS": "カードを数える",
    "GET_GAME_STAT": "ゲーム統計取得",
    "APPLY_MODIFIER": "修正を適用 (継続効果)",
    "REVEAL_CARDS": "カードを公開",
    "REGISTER_DELAYED_EFFECT": "遅延効果登録",
    "RESET_INSTANCE": "カード状態リセット",

    # Scope
    "PLAYER_SELF": "自分",
    "PLAYER_OPPONENT": "相手",
    "TARGET_SELECT": "選択",
    "ALL_PLAYERS": "両者",
    "RANDOM": "ランダム",
    "ALL_FILTERED": "全て(フィルタ)",

    # Zones
    "BATTLE_ZONE": "バトルゾーン",
    "MANA_ZONE": "マナゾーン",
    "HAND": "手札",
    "GRAVEYARD": "墓地",
    "SHIELD_ZONE": "シールドゾーン",
    "DECK": "山札"
}

def tr(text):
    return TRANSLATIONS.get(text, text)
