# Duel Masters AI Simulator - Ultimate Master Specification (v2.0 Final)

## 1. プロジェクト概要 (Project Overview)
- **プロジェクト名**: DM_AI_Simulator
- **目的**: デュエル・マスターズのメタゲーム解明（最強デッキ探索）およびプレイングの最適化。
- **開発哲学**:
    - **High-Speed**: C++20によるビットボード・メモリプール・ゼロコピー転送を駆使した超高速エンジン。
    - **AlphaZero-based**: MCTS + MLPによる自己対戦強化学習。
    - **Evolutionary**: 評価ベースのサイドボード交換によるデッキ進化。
    - **Scientific**: PyQt6による可視化とヒートマップ分析。

## 2. システムアーキテクチャ (System Architecture)

### 2.1 技術スタック
- **Core Engine**: C++20 (GCC/Clang/MSVC) - std::span, concepts, modules (optional)
- **Binding**: Pybind11 (Zero-copy tensor passing) [Q56]
- **AI/ML**: Python 3.10+, PyTorch (CUDA/MPS auto-detect) [Q61]
- **GUI**: PyQt6 (Non-blocking, Polling-based update) [Q76, Q93]
- **Build**: CMake 3.14+
- **Test**: Python pytest for logic, validator.py for CSV integrity [Q83]

### 2.2 ディレクトリ構成 & 名前空間 [Q86]
名前空間はディレクトリ階層に準拠する（例: `dm::engine::ActionGenerator`）。

```text
dm_simulator/
├── CMakeLists.txt              # Build Config
├── config/                     # [Q84] Runtime Configs (rules.json, train_config.yaml)
├── data/                       # Assets
│   ├── cards.csv               # Card DB (Schema: Q50)
│   ├── decks/                  # Deck Files (Format: ID List [Q59])
│   └── logs/                   # [Q46] Debug Logs
├── src/                        # C++ Source (Namespace Hierarchy)
│   ├── core/                   # [dm::core] Dependencies-free
│   │   ├── types.hpp           # Enums, Consts
│   │   ├── card_def.hpp        # struct CardDefinition
│   │   ├── game_state.hpp      # struct GameState
│   │   └── constants.hpp       # [Q92] Fixed Limits
│   ├── engine/                 # [dm::engine] Game Logic
│   │   ├── action_gen/         # ActionGenerator (Strategic Pass)
│   │   ├── flow/               # PhaseManager, WinCondition
│   │   ├── effects/            # EffectResolver (Stack Machine)
│   │   └── mana/               # ManaSystem (Auto-Tap)
│   ├── ai/                     # [dm::ai] ML Bridge
│   │   └── encoders/           # TensorConverter (Hybrid)
│   ├── utils/                  # [dm::utils] RNG, CSV Loader
│   └── python/                 # Pybind11 Interface
├── py_ai/                      # Python AI Modules
│   ├── agent/                  # Network (MLP), MCTS
│   ├── ga/                     # Deck Evolution (Mutation)
│   └── analytics/              # Heatmap Generator
└── gui/                        # PyQt Frontend
    ├── app.py                  # Main Window
    ├── widgets/                # CardWidget, MCTS_Info
    └── deck_builder.py         # [Q68] GUI Deck Editor
```

## 3. コア・データ仕様 (Core Data Specs)

### 3.1 定数と制限 (constants.hpp) [Q92, Q38]
アクション空間の計算基準となる固定値。
- **MAX_HAND_SIZE**: 20
- **MAX_BATTLE_SIZE**: 20
- **MAX_MANA_SIZE**: 20
- **MAX_GRAVE_SEARCH**: 20 (Tensor入力用)
- **TURN_LIMIT**: 100 (強制引き分け) [Q38, Q90]
- **POWER_INFINITY**: 32000

### 3.2 カードデータ構造 (card_def.hpp)
- **ID管理**: CardID (uint16_t). Path A (CSV) & Path B (Logic Code) [Q80].
- **AltCostKeywords**: GZero, RevolutionChange, MachFighter, GStrike 等をフラグ管理 [Q47].
- **Filter Parsing**: CSVロード時に文字列条件（"OPP_TAPPED"等）をID化して保持 [Q50, Q55].
- **Mode Selection**: ModalEffectGroup 構造体による複数選択管理 [Q71].

### 3.3 盤面状態 (game_state.hpp)
- **Determinism**: `std::mt19937` をState内に保持し、シード値による完全再現を保証 [Q69].
- **Incomplete Info**: 学習用Viewでは相手の手札・山札の中身をマスクまたはランダム化 [Q8].
- **Error Handling**: 異常状態で `std::runtime_error` を送出し、即時停止 [Q43, Q87].

## 4. ゲームルール詳細 (Detailed Game Rules)

### 4.1 勝利条件 & 終了処理
- **Direct Attack**: シールド0枚での被弾。
- **Deck Out (Hard)**: 山札の最後の1枚を引いた瞬間に敗北 [Q27].
- **Simultaneous**: 同時敗北は引き分け (Draw) [Q52].
- **Resignation**: MCTS評価値が5%以下で投了 (10%の確率で検証のため続行) [Q98].

### 4.2 フェーズ進行ロジック
- **Mana Phase**: 5ターン目以降、チャージスキップ(PASS)可能 [Q26].
- **Main Phase**:
    - **Strategic Pass**: プレイ可能カードがあってもPASSを選択可（コンボ温存） [Q26補足].
    - **Evolution**: 進化元が複数ある場合、AIが選択 [Q28].
- **Attack Phase**:
    - **Attack Masking**: 攻撃誘導効果以外の対象をマスク [Q25].
    - **G-Strike**: 攻撃権消費として処理。ダイヤモンドカッター等のレイヤー処理適用 [Q40].
    - **Blocker**: NAP（防御側）AIがブロック有無を選択 [Q37].

### 4.3 効果解決エンジン (The Stack)
- **Priority**: AP待機効果 -> NAP待機効果 の順で解決。
- **Simultaneous Triggers**: 同一プレイヤー内の順序はAIが選択 (Action生成) [Q19].
- **Targeting State Machine**: 対象選択は `SELECT_MODE` -> `SELECT_TARGET` ... と順次ステート遷移する [Q74].
- **Search Logic**:
    - 山札の下に置く順序は全探索 [Q48].
    - 同名カードはグルーピングして探索空間圧縮 [Q75].
- **Replacement Effects**: 連鎖しない（置換の置換禁止） [Q18].

## 5. AI & 機械学習仕様 (AI/ML Specs)

### 5.1 ネットワーク構造 [Q95]
- **Backbone**: MLP (多層パーセプトロン) 5層。
- **Flow**: Input (Tensor) -> FC(1024) -> ReLU -> ... -> FC(ActionSize).
- **Update Policy**: 勝率閾値 (Threshold) 更新。勝率55%超えで新モデル採用 [Q78].

### 5.2 入出力マッピング
- **Input Tensor [Q53, Q57]**:
    - **方式**: Zero-Copy (C++がPythonのメモリに直接書き込み)。
    - **形状**: 固定長 (Hybrid Compression).
    - **内容**: グローバル情報 + 自軍(Full) + 敵軍(Masked).
    - **マナ**: 文明別枚数カウントに圧縮。
    - **墓地**: 最新20枚を個別エンコード。
    - **履歴**: 現在盤面のみ [Q35].
- **Action Space (Total approx. 600 dim) [Q89, Q92]**:
    - Flattened Fixed Vector.
    - MANA(20), PLAY(20), ATTACK(420), BLOCK(20), SELECT_TARGET(100+), PASS(1).

### 5.3 MCTS (Monte Carlo Tree Search)
- **Exploration**: ルートノードにディリクレノイズ加算 [Q96].
- **Tree Policy**: ターンごとにツリーリセット（再利用なし） [Q49].
- **Determinization**: 未公開情報をランダム固定してプレイアウト [Q10].

### 5.4 学習ループ [Q100]
- **Replay Buffer**: Hybrid Buffer.
    - 容量の70%を最新データ（Sliding Window）、30%を過去の高評価データ（Golden Games）で構成し、多様性と適応性を両立。
- **Seed Mgmt**: 学習マネージャーによるシード配布 [Q94].
- **Checkpoint**: 10,000ゲームごとにモデル保存 [Q60].

## 6. メタゲーム進化 (Meta-Game Evolution)

### 6.1 デッキ最適化 (Genetic Algorithm)
- **Method**: 評価ベースのサイドボード交換 (Mutation Only) [Q97].
- **Selection Logic**: カードごとの貢献度（勝率・使用率）を統計し、低いカードを優先的に入れ替える [Q99].
- **Validation**: 殿堂レギュレーション（禁止制限）を厳密にチェック [Q77].
- **Matchmaking**: リーグ戦形式（過去の自分＋環境デッキプール） [Q62].

### 6.2 分析アウトプット [Q42, Q67, Q70]
- **Heatmap**: 世代ごとのカード採用率推移をHTMLレポート出力。
- **Logs**: ファイル出力 (`logs/game_xxx.txt`). 勝因・決着ターン・使用キーカードを記録。
- **State Dump**: バグ発生時等の盤面スナップショット保存（シード再現用） [Q69].

## 7. GUI & 開発ツール (Frontend/DevOps)

### 7.1 PyQt6 GUI [Q65, Q66, Q79]
- **Control**: クリックによるステップ実行 (Step-by-step).
- **Visualization**:
    - MCTSの思考結果（各手の勝率予測）をリスト表示。
    - 戦略的パスの意図を可視化。
    - 解決待ち効果（スタック）をリスト表示。
- **Concurrency**: ポーリング方式による非同期更新（UIフリーズ防止） [Q93].

### 7.2 開発補助
- **Deck Builder**: GUI内蔵エディタを段階的に実装 [Q68].
- **Auto-Gen**: テスト用ランダムデッキ生成ツール [Q85].
- **Config**: `config.json` による外部パラメータ管理 [Q84].

## 8. 実装ロードマップ (Step-by-Step Implementation)

### Phase 1: Foundation (基盤構築)
- **Project Setup**: CMakeLists.txt 作成、ディレクトリ構造作成。
- **Core Types**: `types.hpp`, `constants.hpp` 定義。
- **Data Loader**: `csv_loader.cpp` (String Parsing) 実装。
- **Game State**: `game_state.hpp` (Memory optimized) 実装。
- **Tests**: `validator.py` によるデータ整合性チェック。

### Phase 2: Game Logic Implementation (ロジック実装)
- **Phase Flow**: `PhaseManager` (Win condition, Turn limit) 実装。
- **Mana System**: `ManaCalculator` (Color/Cost logic) 実装。
- **Action Gen**: `ActionGenerator` (Masking, Strategic Pass) 実装。
- **Effects**: `EffectResolver` (Stack, Trigger, Mode Selection) 実装。
- **Milestone**: C++のみでランダム対戦がエラーなく完走すること。

### Phase 3: AI Bridge & Python (AI連携)
- **Tensor**: `TensorConverter` (Hybrid encoding) 実装。
- **Binding**: `bindings.cpp` (Pybind11 Zero-copy) 実装。
- **Python Agent**: `network.py` (MLP), `mcts.py` (Determinization) 実装。

### Phase 4: Frontend & Training System (アプリ化)
- **GUI**: `app.py` (PyQt6 Main Window) 実装。
- **Training Loop**: `train.py` (Self-play, Replay Buffer) 実装。
- **Evolution**: `evolve.py` (GA, Evaluation-based swap) 実装。

### Phase 5: Polish & Analytics (仕上げ)
- **Deck Builder**: GUIへの機能統合。
- **Heatmap**: `analyze.py` 実装。
- **Optimization**: プロファイリングとボトルネック解消。