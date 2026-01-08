# -*- coding: utf-8 -*-
# Localized Japanese text.
from typing import Any, Dict, List, Optional, Union
from types import ModuleType

# dm_ai_module may be an optional compiled module; annotate as Optional[ModuleType]
m: Optional[ModuleType] = None
try:
    import dm_ai_module as m  # type: ignore
except ImportError:
    # leave m as None if module not available
    pass

# Base translations
TRANSLATIONS: Dict[Any, str] = {
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
    "Add Command": "コマンドを追加",
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
    "Mekraid": "メクレイド",
    "Friend Burst": "フレンド・バースト",
    "Untap In": "アンタップイン",
    "Meta Counter": "メタカウンター",
    "Power Attacker": "パワーアタッカー",
    "EX-Life": "EX-Life",
    "Mega Last Burst": "超天篇",
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
    "DM AI Simulator": "DM AI シミュレーター",
    "OK": "OK",
    "Cancel": "キャンセル",
    "Mark this card as critical for the deck's strategy.": "このカードをデッキ戦略の重要カードとしてマークします。",
    "Comma-separated list of races (e.g. 'Dragon, Fire Bird')": "カンマ区切りの種族リスト (例: ドラゴン, ファイアー・バード)",
    "Creature power (ignored for Spells)": "クリーチャーパワー (呪文の場合は無視)",
    "Mana cost of the card": "カードのマナコスト",
    "Card type (Creature, Spell, etc.)": "カードタイプ (クリーチャー、呪文など)",
    "Enter card name...": "カード名を入力...",
    "ID": "ID",
    "Warning": "警告",
    "Please select an Effect or Option to add a Command.": "コマンドを追加する効果またはオプションを選択してください。",
    "Please select a Card or Effect group to add an Effect.": "効果を追加するカードまたは効果グループを選択してください。",
    "Failed to save JSON": "JSONの保存に失敗しました",
    "Saved": "保存完了",
    "Cards saved successfully!": "カードを保存しました!",
    "Failed to load JSON": "JSONの読み込みに失敗しました",
    "Force update the card preview": "プレビューを強制更新",
    "Update Preview": "プレビュー更新",
    "Delete the selected item": "選択した項目を削除",
    "Delete Item": "削除",
    "Add a command to the selected effect": "選択した効果にコマンドを追加",
    "Add a new effect to the selected card": "選択したカードに効果を追加",
    "Add Effect": "効果追加",
    "Save all changes to JSON": "変更をJSONに保存",
    "Save JSON": "JSONを保存",
    "Create a new card": "新しいカードを作成",
    "Card Editor Ver 2.0": "カードエディタ Ver 2.0",
    "Error: dm_ai_module not loaded.": "エラー: dm_ai_module が読み込まれていません。",
    "Close this dialog": "このダイアログを閉じます",
    "Start the batch simulation with current settings": "現在の設定で一括シミュレーションを開始します",
    "Monte Carlo Tree Search simulations per move": "1手あたりのモンテカルロ木探索シミュレーション回数",
    "Number of CPU threads to use": "使用するCPUスレッド数",
    "Total number of games to simulate": "シミュレートするゲームの総数",
    "Select the AI agent type for evaluation": "評価用のAIエージェントタイプを選択してください",
    "Select the game scenario to simulate": "シミュレートするゲームシナリオを選択してください",
    "Select Model File": "モデルファイル選択",
    "Run Simulation": "シミュレーション実行",
    "Note: High simulation counts may cause memory issues (std::bad_alloc).": "注: シミュレーション数が多いとメモリ不足 (std::bad_alloc) が発生する可能性があります。",
    "MCTS Sims": "MCTSシミュレーション数",
    "Threads": "スレッド数",
    "Games": "ゲーム数",
    "Evaluator": "評価器",
    "Scenario": "シナリオ",
    "Settings": "設定",
    "Batch Simulation / Verification": "一括シミュレーション / 検証",
    "Throughput": "スループット",
    "Draws": "引き分け",
    "Losses": "敗北",
    "Wins": "勝利",
    "Completed": "完了",
    "Simulation Error": "シミュレーションエラー",
    "Starting simulation": "シミュレーションを開始します",
    "Using initialized model (Untrained)": "初期化済みモデル (未学習) を使用します",
    "Initializing...": "初期化中...",
    "Info": "情報",
    "Success": "成功",
    "Failed to reload database: {e}": "データベース再読み込み失敗: {e}",
    "Database reloaded!": "データベースを再読み込みしました!",
    "Reload DB": "DB再読込",
    "Refresh": "更新",
    "Resolve Selected": "選択を解決",
    "Pending Effects (Stack)": "待機中の効果 (スタック)",
    "勝率予測: %p%": "勝率予測: %p%",
    "グラフ表示": "グラフ表示",
    "テーブル表示": "テーブル表示",
    "優先度 (P)": "優先度 (P)",
    "評価値 (Q)": "評価値 (Q)",
    "訪問回数": "訪問回数",
    "アクション": "アクション",
    "AI思考プロセス (MCTS)": "AI思考プロセス (MCTS)",
    "(Cost: {cost})": "(コスト: {cost})",
    "Unknown Item": "不明なアイテム",
    "Pending Effects": "待機中の効果",
    "dm_ai_module not found. Please build the C++ extension.": "dm_ai_module が見つかりません。C++拡張をビルドしてください。",
    "Confirm": "確定",
    "Invalid Selection": "無効な選択",
    "Please select at least {min_targets} target(s).": "少なくとも {min_targets} 枚の対象を選択してください。",
    "Scenario Mode Disabled": "シナリオモード無効化",
    "Scenario Mode Enabled": "シナリオモード有効化",
    "Error reloading cards": "カード読み込みエラー",
    "Card Data Reloaded from Editor Save": "カードデータを再読み込みしました",
    "Scenario Editor not available.": "シナリオエディタは利用できません。",
    "Select effect to resolve:": "解決する効果を選択してください:",
    "Select Trigger": "トリガー選択",
    "Please select cards:": "カードを選択してください:",
    "Select Cards": "カード選択",
    "Choose an option:": "オプションを選択してください:",
    "Select Option": "オプション選択",
    "Main Toolbar": "メインツールバー",
    "Ready": "準備完了",
    "Recording Loop...": "ループ記録中...",
    "Start Recording": "記録開始",
    "Stop & Verify": "停止＆検証",
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
    "Action Group": "アクショングループ",
    "Command Type": "アクションタイプ",
    "Mutation Kind": "変異種別 (Mutation Kind)",
    "String Param": "文字列パラメータ",
    "Amount": "量/数値",
    "From Zone": "移動元ゾーン",
    "To Zone": "移動先ゾーン",
    "Optional (Arbitrary Amount)": "任意 (任意数)",
    "Query Mode": "クエリモード",
    "TRANSITION": "カード移動",
    "MUTATE": "状態変更",
    "FLOW": "進行制御",
    "QUERY": "カード情報取得",
    "DECIDE": "決定",
    "DECLARE_REACTION": "リアクション宣言",
    "MOVE": "移動 (Move)",
    "DRAW_PLAY": "ドロー/プレイ (Draw/Play)",
    "GRANT": "付与 (Grant)",
    "CONTROL": "制御 (Control)",
    "OTHER": "その他 (Other)",
    # 新コマンドグループ分類
    "DRAW": "ドロー",
    "CARD_MOVE": "カード移動",
    "DECK_OPS": "デッキ操作",
    "PLAY": "プレイ",
    "CHEAT_PUT": "踏み倒し",
    "GRANT": "付与（状態変更）",
    "LOGIC": "ロジック",
    "BATTLE": "バトル",
    "RESTRICTION": "制限",
    "SPECIAL": "特殊",
    "This type maps to a native Command. It will be saved as a Command.": "このタイプはネイティブコマンドに対応しています。コマンドとして保存されます。",
    "Convert to Command": "コマンドに変換",
    "Source Zone": "移動元ゾーン",
    "Allow Duplicates": "重複を許可",
    "COST_MODIFIER": "コスト修正",
    "POWER_MODIFIER": "パワー修正",
    "GRANT_KEYWORD": "キーワード付与",
    "SET_KEYWORD": "キーワード設定",
    "Logic Tree": "ロジックツリー",
    "Cards to Draw": "引く枚数",
    "Count": "枚数",
    "Count (if selecting)": "枚数（選択する場合）",
    "Mutation Type": "変異種別",
    "Value / Duration": "値／持続ターン数",
    "Extra Param": "追加パラメータ",
    "Power Adjustment": "パワー調整値",
    "Query Result": "クエリ結果",
    "Flow Instruction": "フロー指示",
    "Found Cards": "見つかったカード",
    "Look Count": "参照枚数",
    "Level": "レベル",
    "Played Card": "プレイしたカード",
    "Max Cost": "最大コスト",
    "Hint": "ヒント",
    "Token ID": "トークンID",
    "Selected Number": "選択した数",
    "Generate Options": "選択肢を生成",
    "Options to Add": "追加する選択肢数",
    "Manual Input": "手動入力",
    "Property Inspector": "プロパティインスペクタ",
    "Select an item to edit": "編集する項目を選択してください",
    "Stat Key": "統計キー",
    "String": "文字列",
    "Preset: Mana Civs": "プリセット: マナ文明",
    "Set stat key to MANA_CIVILIZATION_COUNT": "統計キーを MANA_CIVILIZATION_COUNT に設定",
    "CARDS_MATCHING_FILTER": "フィルタ一致カード",
    "MANA_CIVILIZATION_COUNT": "マナ文明数",
    "SHIELD_COUNT": "シールド枚数",
    "HAND_COUNT": "手札枚数",
    "CARDS_DRAWN_THIS_TURN": "このターンに引いた枚数",
    "SYM_CREATURE": "参照: クリーチャー",
    "SYM_SPELL": "参照: 呪文",
    "G_ZERO": "G・ゼロ",
    "HYPER_ENERGY": "ハイパーエナジー",
    "STAT": "統計更新",
    "GAME_RESULT": "ゲーム終了",
    "POWER_MOD": "パワー修正",
    "ADD_KEYWORD": "キーワード付与",
    "MANA_CHARGE": "マナゾーンに置く",
    "PHASE_CHANGE": "フェーズ移行",
    "TURN_CHANGE": "ターン変更",
    "SET_ACTIVE_PLAYER": "手番変更",
    "CARDS_DRAWN": "ドロー枚数",
    "CARDS_DISCARDED": "手札破棄枚数",
    "CREATURES_PLAYED": "クリーチャープレイ数",
    "SPELLS_CAST": "呪文詠唱数",
    "P1_WIN": "P1勝利",
    "P2_WIN": "P2勝利",
    "DRAW": "引き分け",
    "NONE": "なし",
    "PLAYER_SELF": "自分",
    "PLAYER_OPPONENT": "相手",
    "CANNOT_ATTACK": "攻撃できない",
    "CANNOT_BLOCK": "ブロックできない",
    "CANNOT_ATTACK_OR_BLOCK": "攻撃もブロックもできない",

    # --- Fallback Enum Translations (No dm_ai_module) ---
    # Civilizations
    "LIGHT": "光",
    "WATER": "水",
    "DARKNESS": "闇",
    "FIRE": "火",
    "NATURE": "自然",
    "ZERO": "ゼロ",
    "COLORLESS": "無色",

    # Zones (JSON / editor)
    "BATTLE_ZONE": "バトルゾーン",
    "MANA_ZONE": "マナゾーン",
    "SHIELD_ZONE": "シールドゾーン",
    "HAND": "手札",
    "GRAVEYARD": "墓地",
    "DECK": "山札",
    "DECK_TOP": "山札の上",
    "DECK_BOTTOM": "山札の下",

    # Zones (Command / native-style)
    "BATTLE": "バトルゾーン",
    "MANA": "マナゾーン",
    "SHIELD": "シールドゾーン",

    # Common triggers (if leaked)
    "ON_PLAY": "出た時",
    "AT_ATTACK": "攻撃する時",
    "ON_DESTROY": "破壊された時",
    "AT_END_OF_TURN": "ターンの終わりに",
    "TURN_START": "ターンのはじめに",
    "S_TRIGGER": "S・トリガー",

    # Command Types (Macros)
    "DRAW_CARD": "カードを引く",
    "DISCARD": "手札を捨てる",
    "DESTROY": "破壊",
    "TAP": "タップ",
    "UNTAP": "アンタップ",
    "RETURN_TO_HAND": "手札に戻す",
    "BREAK_SHIELD": "シールドブレイク",
    "SEARCH_DECK": "デッキ探索",
    "SHIELD_TRIGGER": "S・トリガー",

    # Editor Actions (Legacy)
    "COST_REDUCTION": "コスト軽減",
    "MEASURE_COUNT": "カウント/計測",

    # Tooltips
    "Open the Deck Builder tool": "デッキビルダーを開きます",
    "Open the Card Editor tool": "カードエディタを開きます",
    "Toggle Scenario Mode for testing specific game states": "シナリオモードの切り替え (特定盤面のテスト)",
    "Run multiple games for statistical analysis": "統計分析用の一括シミュレーションを実行",
    "Toggle the MCTS Analysis dock": "MCTS分析ドックの表示切り替え",
    "Toggle the Loop Recorder dock": "ループ記録ドックの表示切り替え",
    "Start/Stop continuous simulation": "シミュレーションの開始/停止",
    "Advance game by one step": "ゲームを1ステップ進める",
    "Confirm target selection": "対象選択を確定",
    "Reset the game state": "ゲーム状態をリセット",

    # Messages with placeholders
    "Loaded scenario: {name}": "シナリオを読み込みました: {name}",
    "Scenario '{name}' saved.": "シナリオ '{name}' を保存しました。",
    "Failed to add card: {error}": "カード追加に失敗しました: {error}",
    "Failed to save: {error}": "保存に失敗しました: {error}",
    "Include {zone} in target selection": "{zone} を対象に含める",
    "This type '{cmd_type}' is only supported by the Legacy Action format.": "このタイプ '{cmd_type}' はレガシーAction形式のみ対応です。",
    "Warning: Imperfect Conversion from {orig}": "警告: {orig} からの変換は不完全な可能性があります",
    "Race": "種族",
    "[Creature]": "[クリーチャー]",
    "Creature Name": "クリーチャー名",
    "Creature Text": "クリーチャーテキスト",
    "Spell Name": "呪文名",
    "Spell Text": "呪文テキスト",

    # Card Preview Pane
    "Card Preview": "カードプレビュー",
    "Generated Text (Source):": "生成テキスト（元データ）:",
    "CIR Summary:": "CIRサマリー:",
    "Effect": "効果",
    "Command": "コマンド",
    "Action": "アクション",
    "Legacy": "レガシー",

    # Phase 2 Main Screen Keys (Zones)
    "P0 Hand": "P0 手札",
    "P0 Mana": "P0 マナ",
    "P0 Graveyard": "P0 墓地",
    "P0 Battle Zone": "P0 バトルゾーン",
    "P0 Shield Zone": "P0 シールド",
    "P0 Deck": "P0 デッキ",
    "P1 Hand": "P1 手札",
    "P1 Mana": "P1 マナ",
    "P1 Graveyard": "P1 墓地",
    "P1 Battle Zone": "P1 バトルゾーン",
    "P1 Shield Zone": "P1 シールド",
    "P1 Deck": "P1 デッキ",

    # Phases
    "Start Phase": "開始フェーズ",
    "Draw Phase": "ドローフェーズ",
    "Mana Phase": "マナフェーズ",
    "Main Phase": "メインフェーズ",
    "Attack Phase": "攻撃フェーズ",
    "Block Phase": "ブロックフェーズ",
    "End Phase": "終了フェーズ",

    # ZoneWidget
    "Deck ({count})": "デッキ ({count})",
    "Shield ({count})": "シールド ({count})",

    # CardDetailPanel
    "Civ": "文明",

    # App Log
    "Execution Error": "実行エラー",
    "Execution Error: {error}": "実行エラー: {error}",
    "Turn: {turn}": "ターン: {turn}",
    "Phase: {phase}": "フェーズ: {phase}",
    "Active: P{player_id}": "アクティブ: P{player_id}",
    "Loaded Deck for P0": "P0のデッキを読み込みました",
    "Loaded Deck for P1": "P1のデッキを読み込みました",

    # Loop recorder / logs
    "Start Hash: {hash}": "開始ハッシュ: {hash}",
    "End Hash: {hash}": "終了ハッシュ: {hash}",
    "Resources: Hand={hand}, Mana={mana}": "リソース: 手札={hand}, マナ={mana}",
    "Action: {action}": "アクション: {action}",
    "State Match: YES (Exact Hash)": "状態一致: はい (完全一致)",
    "State Match: NO": "状態一致: いいえ",
    "Hand Diff: {diff}": "手札差分: {diff}",
    "Mana Diff: {diff}": "マナ差分: {diff}",
    "RESULT: Infinite Loop with Advantage Proven!": "結果: 無限ループ + 有利が証明されました!",
    "RESULT: Loop Proven (No Resource Gain detected yet)": "結果: ループは証明されました (リソース増加は未検出)",
    "Game Initialized via Controller": "ゲーム初期化 (Controller)",
    "Game Over": "ゲーム終了",
    "Controller Action: {type}": "Controllerアクション: {type}",
    "Controller Execution Error: {error}": "Controller実行エラー: {error}",

    # MCTS tooltips
    "Action: {name}\nVisits: {visits}\nValue: {value}": "アクション: {name}\n訪問回数: {visits}\n評価値: {value}",

    # Editor tooltips
    "Legacy Action: {orig}\nPlease replace with modern Commands.": "レガシーAction: {orig}\n可能なら新しいCommandに置き換えてください。",
}

# Add Enum mappings if module is available
if m:
    # EffectActionType (only if available on the module)
    if hasattr(m, 'EffectActionType'):
        _effect_map = {
            'GRANT_KEYWORD': "キーワード付与",
            'MOVE_CARD': "カード移動",
            'FRIEND_BURST': "フレンド・バースト",
            'APPLY_MODIFIER': "効果付与",
            'DRAW_CARD': "カードを引く",
            'ADD_MANA': "マナ追加",
            'DESTROY': "破壊",
            'RETURN_TO_HAND': "手札に戻す",
            'SEND_TO_MANA': "マナ送りにする",
            'TAP': "タップする",
            'UNTAP': "アンタップする",
            'MODIFY_POWER': "パワー修正",
            'BREAK_SHIELD': "シールドブレイク",
            'LOOK_AND_ADD': "見て加える(サーチ)",
            'SEARCH_DECK_BOTTOM': "デッキ下サーチ",
            'MEKRAID': "メクレイド",
            'REVOLUTION_CHANGE': "革命チェンジ",
            'COUNT_CARDS': "カードカウント",
            'GET_GAME_STAT': "ゲーム統計取得",
            'REVEAL_CARDS': "カード公開",
            'RESET_INSTANCE': "カード状態リセット",
            'REGISTER_DELAYED_EFFECT': "遅延効果登録",
            'SEARCH_DECK': "デッキ探索",
            'SHUFFLE_DECK': "デッキシャッフル",
            'ADD_SHIELD': "シールド追加",
            'SEND_SHIELD_TO_GRAVE': "シールド焼却",
            'SEND_TO_DECK_BOTTOM': "デッキ下に送る",
            'MOVE_TO_UNDER_CARD': "カードの下に重ねる",
            'CAST_SPELL': "呪文を唱える",
            'PUT_CREATURE': "クリーチャーを出す",
            'COST_REFERENCE': "コスト参照/軽減",
            'SELECT_NUMBER': "数字を選択",
            'SUMMON_TOKEN': "トークン生成",
            'DISCARD': "手札を捨てる",
            'PLAY_FROM_ZONE': "ゾーンからプレイ",
            'LOOK_TO_BUFFER': "バッファへ移動(Look)",
            'SELECT_FROM_BUFFER': "バッファから選択",
            'PLAY_FROM_BUFFER': "バッファからプレイ",
            'MOVE_BUFFER_TO_ZONE': "バッファから移動",
            'SELECT_OPTION': "選択肢",
            'RESOLVE_BATTLE': "バトル解決",
        }
        for _name, _text in _effect_map.items():
            _member = getattr(m.EffectActionType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # ActionType (map by name to avoid referencing missing enum attributes)
    if hasattr(m, 'ActionType'):
        _action_map = {
            'PLAY_CARD': "カードをプレイ",
            'ATTACK_CREATURE': "クリーチャー攻撃",
            'ATTACK_PLAYER': "プレイヤー攻撃",
            'BLOCK': "ブロック",
            'USE_SHIELD_TRIGGER': "S・トリガー使用",
            'RESOLVE_EFFECT': "効果解決",
            'RESOLVE_PLAY': "プレイ解決",
            'DECLARE_PLAY': "プレイ宣言",
            'SELECT_TARGET': "対象選択",
            'USE_ABILITY': "能力使用",
            'DECLARE_REACTION': "リアクション宣言",
            'MANA_CHARGE': "マナゾーンに置く",
            'PAY_COST': "コスト支払い",
            'PASS': "パス",
        }
        for _name, _text in _action_map.items():
            _member = getattr(m.ActionType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # TriggerType
    if hasattr(m, 'TriggerType'):
        _trigger_map = {
            'ON_PLAY': "出た時 (CIP)",
            'ON_ATTACK': "攻撃する時",
            'ON_DESTROY': "破壊された時",
            'ON_OPPONENT_DRAW': "相手がドローした時",
            'S_TRIGGER': "S・トリガー",
            'TURN_START': "ターン開始時",
            'PASSIVE_CONST': "常在効果(パッシブ)",
        }
        for _name, _text in _trigger_map.items():
            _member = getattr(m.TriggerType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Civilization
    if hasattr(m, 'Civilization'):
        _civ_map = {
            'FIRE': "火",
            'WATER': "水",
            'NATURE': "自然",
            'LIGHT': "光",
            'DARKNESS': "闇",
            'ZERO': "無色",
        }
        for _name, _text in _civ_map.items():
            _member = getattr(m.Civilization, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Zone
    if hasattr(m, 'Zone'):
        _zone_map = {
            'HAND': "手札",
            'BATTLE': "バトルゾーン",
            'GRAVEYARD': "墓地",
            'MANA': "マナゾーン",
            'SHIELD': "シールドゾーン",
            'DECK': "デッキ",
            'BUFFER': "バッファ",
            'UNDER_CARD': "カードの下",
        }
        for _name, _text in _zone_map.items():
            _member = getattr(m.Zone, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # TargetScope
    if hasattr(m, 'TargetScope'):
        _ts_map = {
            'SELF': "自分",
            'TARGET_SELECT': "対象選択",
            'NONE': "なし",
        }
        for _name, _text in _ts_map.items():
            _member = getattr(m.TargetScope, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Command Types
    if hasattr(m, 'CommandType'):
        _cmd_map = {
            'TRANSITION': "カード移動",
            'MUTATE': "状態変更",
            'FLOW': "進行制御",
            'QUERY': "カード情報取得",
            'DECIDE': "決定",
            'DECLARE_REACTION': "リアクション宣言",
            'STAT': "統計更新",
            'GAME_RESULT': "ゲーム終了",
        }
        for _name, _text in _cmd_map.items():
            _member = getattr(m.CommandType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Flow Types
    if hasattr(m, 'FlowType'):
        _flow_map = {
            'PHASE_CHANGE': "フェーズ移行",
            'TURN_CHANGE': "ターン変更",
            'SET_ACTIVE_PLAYER': "手番変更",
        }
        for _name, _text in _flow_map.items():
            _member = getattr(m.FlowType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Mutation Types
    if hasattr(m, 'MutationType'):
        _mut_map = {
            'TAP': "タップ",
            'UNTAP': "アンタップ",
            'POWER_MOD': "パワー修正",
            'ADD_KEYWORD': "キーワード付与",
            'REMOVE_KEYWORD': "キーワード削除",
            'ADD_PASSIVE_EFFECT': "パッシブ効果付与",
            'ADD_COST_MODIFIER': "コスト修正付与",
            'ADD_PENDING_EFFECT': "待機効果追加",
        }
        for _name, _text in _mut_map.items():
            _member = getattr(m.MutationType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Stat Types
    if hasattr(m, 'StatType'):
        _stat_map = {
            'CARDS_DRAWN': "ドロー枚数",
            'CARDS_DISCARDED': "手札破棄枚数",
            'CREATURES_PLAYED': "クリーチャープレイ数",
            'SPELLS_CAST': "呪文詠唱数",
        }
        for _name, _text in _stat_map.items():
            _member = getattr(m.StatType, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Game Result
    if hasattr(m, 'GameResult'):
        _gr_map = {
            'NONE': "なし",
            'P1_WIN': "P1勝利",
            'P2_WIN': "P2勝利",
            'DRAW': "引き分け",
        }
        for _name, _text in _gr_map.items():
            _member = getattr(m.GameResult, _name, None)
            if _member is not None:
                TRANSLATIONS[_member] = _text

    # Also keep string keys for Enums for backward compatibility or serialization
    enum_candidates = [
        getattr(m, 'ActionType', None),
        getattr(m, 'EffectActionType', None),
        getattr(m, 'TriggerType', None),
        getattr(m, 'Civilization', None),
        getattr(m, 'Zone', None),
        getattr(m, 'TargetScope', None),
        getattr(m, 'CommandType', None),
        getattr(m, 'FlowType', None),
        getattr(m, 'MutationType', None),
        getattr(m, 'StatType', None),
        getattr(m, 'GameResult', None),
    ]
    for enum_cls in [e for e in enum_candidates if e is not None]:
        for member in enum_cls.__members__.values():
            if member in TRANSLATIONS:
                TRANSLATIONS[member.name] = TRANSLATIONS[member]

def translate(key: Any) -> str:
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

def tr(text: Any) -> str:
    return translate(text)

def get_card_civilizations(card_data: Any) -> List[str]:
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

def get_card_civilization(card_data: Any) -> str:
    """
    Returns the primary civilization name as a string.
    If multiple, returns the first one.
    """
    civs = get_card_civilizations(card_data)
    if civs:
        return civs[0]
    return "COLORLESS"

def get_card_name_by_instance(game_state: Any, card_db: Dict[int, Any], instance_id: int) -> str:
    if not game_state or not m: return f"Inst_{instance_id}"

    try:
        # Assuming GameState has get_card_instance exposed
        inst = game_state.get_card_instance(instance_id)
        if inst:
            card_id = inst.card_id
            if card_id in card_db:
                return card_db[card_id].name  # type: ignore
    except Exception:
        pass

    return f"Inst_{instance_id}"

def describe_command(cmd: Any, game_state: Any, card_db: Any) -> str:
    """Generate a localized string description for a GameCommand."""
    if not m:
        return "GameCommand（ネイティブモジュール未ロード）"

    cmd_type = cmd.get_type()

    if cmd_type == m.CommandType.TRANSITION:
        # TransitionCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.card_instance_id)
        return f"[{tr('TRANSITION')}] {name} (P{c.owner_id}): {tr(c.from_zone)} -> {tr(c.to_zone)}"

    elif cmd_type == m.CommandType.MUTATE:
        # MutateCommand
        c = cmd
        name = get_card_name_by_instance(game_state, card_db, c.target_instance_id)
        mutation = tr(c.mutation_type)
        val = ""
        if c.mutation_type == m.MutationType.POWER_MOD:
            val = f"{c.int_value:+}"
        elif c.mutation_type == m.MutationType.ADD_KEYWORD:
            val = c.str_value

        return f"[{tr('MUTATE')}] {name}: {mutation} {val}".strip()

    elif cmd_type == m.CommandType.FLOW:
        # FlowCommand
        c = cmd
        flow = tr(c.flow_type)
        val = c.new_value
        if c.flow_type == m.FlowType.PHASE_CHANGE:
            # Cast int to Phase enum if possible
            try:
                val = tr(m.Phase(c.new_value))
            except:
                pass
        return f"[{tr('FLOW')}] {flow}: {val}"

    elif cmd_type == m.CommandType.QUERY:
        c = cmd
        return f"[{tr('QUERY')}] {tr(c.query_type)}"

    elif cmd_type == m.CommandType.DECIDE:
        c = cmd
        return f"[{tr('DECIDE')}] 選択肢: {c.selected_option_index}, 対象数: {len(c.selected_indices)}"

    elif cmd_type == m.CommandType.STAT:
        c = cmd
        return f"[{tr('STAT')}] {tr(c.stat)} += {c.amount}"

    elif cmd_type == m.CommandType.GAME_RESULT:
        c = cmd
        return f"[{tr('GAME_RESULT')}] {tr(c.result)}"

    return f"未対応コマンド: {cmd_type}"
