# -*- coding: utf-8 -*-

# Japanese Localization Dictionary
# Format: "KEY": "Japanese Text"

LOCALIZATION = {
    # Menu
    "MENU_FILE": "ファイル",
    "MENU_EXIT": "終了",
    "MENU_TOOLS": "ツール",
    "MENU_CARD_EDITOR": "カードエディタ",
    "MENU_SCENARIO_EDITOR": "シナリオエディタ",
    "MENU_SIMULATION": "シミュレーション",
    "MENU_HELP": "ヘルプ",
    "MENU_ABOUT": "バージョン情報",

    # Common
    "BTN_OK": "OK",
    "BTN_CANCEL": "キャンセル",
    "BTN_SAVE": "保存",
    "BTN_LOAD": "読み込み",
    "BTN_ADD": "追加",
    "BTN_REMOVE": "削除",
    "BTN_APPLY": "適用",
    "BTN_CLOSE": "閉じる",

    # Card Editor
    "TITLE_CARD_EDITOR": "カードエディタ Ver 2.0",
    "LBL_ID": "ID",
    "LBL_NAME": "名前",
    "LBL_COST": "コスト",
    "LBL_CIVILIZATION": "文明",
    "LBL_TYPE": "タイプ",
    "LBL_POWER": "パワー",
    "LBL_RACES": "種族",
    "LBL_EFFECTS": "効果",
    "LBL_KEYWORDS": "キーワード能力",
    "TAB_BASIC": "基本情報",
    "TAB_LOGIC": "ロジック",
    "TAB_SOURCE": "ソース",
    "GRP_HIERARCHY": "構造",
    "GRP_PROPERTIES": "プロパティ",
    "BTN_ADD_EFFECT": "効果追加",
    "BTN_ADD_ACTION": "アクション追加",
    "BTN_ADD_CONDITION": "条件追加",

    # Filter Editor
    "LBL_FILTER_ZONES": "対象ゾーン",
    "LBL_FILTER_CIVS": "文明フィルタ",
    "LBL_FILTER_RACES": "種族フィルタ",
    "LBL_FILTER_COST": "コスト範囲",
    "LBL_FILTER_POWER": "パワー範囲",
    "LBL_FILTER_FLAGS": "フラグ条件 (タップ/ブロッカー等)",
    "LBL_FILTER_OWNER": "所有者",
    "VAL_PLAYER_SELF": "自分",
    "VAL_PLAYER_OPPONENT": "相手",
    "VAL_PLAYER_BOTH": "両方",

    # Simulation Dialog
    "TITLE_SIMULATION": "一括シミュレーション",
    "GRP_SETTINGS": "設定",
    "LBL_EPISODES": "エピソード数 (対戦数)",
    "LBL_THREADS": "スレッド数",
    "LBL_BATCH_SIZE": "バッチサイズ",
    "LBL_MCTS_SIMS": "MCTSシミュレーション回数",
    "LBL_DECK_1": "デッキ1 (自分)",
    "LBL_DECK_2": "デッキ2 (相手)",
    "BTN_SELECT_FILE": "ファイル選択",
    "BTN_RUN": "実行",
    "BTN_STOP": "停止",
    "GRP_LOG": "実行ログ",
    "MSG_SIM_START": "シミュレーションを開始します...",
    "MSG_SIM_FINISHED": "シミュレーション終了",
    "MSG_SIM_STOPPED": "シミュレーションを停止しました",
    "SIM_MODE": "シミュレーションモード",
    "SIM_MODE_1V1": "1 vs 1",
    "SIM_MODE_LEAGUE": "リーグ戦 (総当たり)",
    "SIM_PBT_ENABLE": "PBT有効化",
    "SIM_DECK_FOLDER": "デッキフォルダ選択",
    "SIM_TAB_LOG": "ログ・グラフ",
    "SIM_TAB_ANALYSIS": "カード分析",
    "COL_CARD_NAME": "カード名",
    "COL_ADOPT_WIN": "採用時勝率",
    "COL_PLAY_WIN": "場に出た時の勝率",
    "COL_GAMES": "採用回数",

    # Scenario Editor
    "TITLE_SCENARIO_EDITOR": "シナリオエディタ",
    "LBL_SCENARIO_LIST": "シナリオ一覧",
    "LBL_SCENARIO_NAME": "シナリオ名",
    "LBL_PLAYER_SETUP": "プレイヤー設定",
    "LBL_OPPONENT_SETUP": "相手設定",
    "LBL_HAND": "手札",
    "LBL_MANA": "マナゾーン",
    "LBL_BATTLE": "バトルゾーン",
    "LBL_SHIELDS": "シールド",
    "LBL_GRAVE": "墓地",
    "LBL_DECK_COUNT": "デッキ枚数",
    "BTN_UPDATE_JSON": "JSON更新",
}

def get_text(key):
    return LOCALIZATION.get(key, key)
