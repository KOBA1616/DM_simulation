# 最強デッキ作成AI プロジェクト設計書
## ～ C++エンジンと結果スタッツによる自律進化システム ～

## 1. プロジェクト概要
本プロジェクトは、トレーディングカードゲーム（TCG）において、人間が与えた定石やタグに頼らず、AIが自律的に「カードの強さ」を数値化し、「最強のデッキ」と「最強のプレイング」を同時に学習するシステムを構築するものである。

### コアコンセプト
*   **PC電源OFF・無料**: Kaggle Notebooks等のクラウド資源を活用し、24時間学習を継続する。
*   **Result Stats (結果スタッツ)**: テキスト解析を行わず、対戦シミュレーションの結果（事実）のみからカード性能を定義する。
*   **Hybrid Architecture**: 高速なC++エンジンと、柔軟なPython強化学習モデルを融合させる。

## 2. インフラ構成 (PC電源OFF・無料運用)
### 2.1 実行環境: Kaggle Scheduled Notebooks
*   **役割**: 計算リソース（GPU/CPU）の提供。
*   **仕様**: 週30時間以上のGPU枠、またはTPU枠を使用。
*   **自動化**: Scheduled Run 機能により、PCの電源を切っていても毎日定時に起動し、最大9時間の学習を行う。

### 2.2 データ永続化: Kaggle Datasets / Hugging Face
*   **役割**: 学習済みモデル、生成されたスタッツデータ、デッキリストの保存。
*   **バージョン管理**:
    *   `model_v{N}.pth`: 学習済みモデル（脳）。
    *   `stats_v{N}.bin`: カードの統計データ（知識）。
    *   `deck_history.json`: 生成されたデッキの変遷。
*   **リレー方式**: 起動時に前回のDatasetからデータをロードし、学習終了時に新しいVersionとして保存・更新する。

## 3. カードデータ定義 (Card Representation)
AIがカードを理解するためのデータ構造。人間によるタグ付けを廃止し、**「基本スペック」と「行動履歴（スタッツ）」**の2層で構成する。

### 3.1 第1層: 基本パラメータ (Static Specs)
ルール処理に必要な、不変の物理スペック。

| パラメータ | 型 | 説明 | AIへの入力意味 |
|---|---|---|---|
| **Original Cost** | Int | 表記コスト | カードの「格（パワーレベル）」 |
| **Min Cost** | Int | 理論最小コスト | Gゼロなら0。デッキタイプの推論に使用 |
| **Power** | Int | パワー数値 | 盤面制圧力 |
| **Civilization** | Bitmask | 文明フラグ | デッキカラー（戦術傾向）の判定 |
| **Type** | Enum | カード種類 | クリーチャー/呪文/城などの判別 |
| **Keywords** | Flags | SA/ブロッカー/トリガー等 | 攻撃・防御・カウンター性能 |

### 3.2 第2層: 結果スタッツ (Behavioral Stats) - 16次元
シミュレーション結果から自動算出される動的スペック。本システムの核。
カード1枚につき、以下の16個の浮動小数点数（Float）を持つ。

| No. | カテゴリ | 項目名 | 定義・計算式 | AIの解釈 |
|---|---|---|---|---|
| 0 | **Timing** | Early Usage | 1~3ターン目の使用率 | 初動・マナ加速 |
| 1 | | Late Usage | 7ターン目以降の使用率 | フィニッシャー |
| 2 | | Trigger Rate | 相手ターン中の使用率 | シールド・トリガー（防御札） |
| 3 | | Cost Discount | (表記コスト) - (実コスト) | コスト軽減・踏み倒し性能 |
| 4 | **Advantage** | Hand Adv | (自手札増 - 敵手札増) | ドローソース / ハンデス |
| 5 | | Board Adv | (自盤面増 - 敵盤面減) | 展開 / 除去 |
| 6 | | Mana Adv | マナゾーン増加数 | マナ加速性能 |
| 7 | | Shield Dmg | 相手シールド減少数 | ブレイク性能・攻撃力 |
| 8 | **Risk** | Hand Var | 手札増減の分散 | 不確定要素（ランダムハンデス等） |
| 9 | | Board Var | 盤面増減の分散 | 全体除去等の状況依存性 |
| 10 | | Survival Rate | 次ターン生存率 | 除去耐性・場持ちの良さ |
| 11 | | Effect Death | 効果破壊された割合 | パワー不足・呪文への脆さ |
| 12 | **Impact** | Win Rate | 使用時の勝率偏差 | 単純なカードパワー |
| 13 | | Comeback | 劣勢時使用の勝率 | 逆転要素（トリガー・革命） |
| 14 | | Finish Blow | 出したターンに勝利 | リーサルウェポン（SA） |
| 15 | | Deck Size | 平均山札消費枚数 | ライブラリアウト戦術適正 |

## 4. 学習サイクル (The Loop)
### Phase 1: データ収集 (Observer)
*   **エンジン**: C++ (High Speed Mode)
*   **対戦**: ランダムデッキ vs ランダムデッキ (100万回)
*   **目的**: 偏見のない初期スタッツ (`stats_v1.bin`) を生成する。
*   **成果**: コンボカードは低評価、単体パワーカードは高評価される。

### Phase 2: AI育成 (Learner)
*   **エンジン**: Python (PPO / Reinforcement Learning)
*   **入力**: `[盤面情報] + [現在のスタッツ (16dim)] + [動的コスト]`
*   **対戦**:
    *   **Tier 1**: 過去の最強モデル (Hall of Fame)
    *   **Tier 2**: 既存のメタデッキ (固定ロジック)
    *   **Tier 3**: 特定コンボ特化AI
*   **目的**: スタッツ情報を頼りに、勝てるプレイングとデッキ構築を学ぶ。

### Phase 3: データの純化 (Refinement)
*   **エンジン**: C++
*   **対戦**: 学習済みAIモデル同士の対戦 (10万回)
*   **目的**: 「賢くなったAI」が使うことで、コンボカードやシナジーカードの真価を発揮させ、スタッツ (`stats_v2.bin`) を更新する。
*   **成果**: 「弱い」と判定されていたコンボパーツが「勝率貢献度高」に修正される。

## 5. デッキ構築ロジック (Meta-Game Optimization)
AIはプレイングだけでなく、デッキの中身も最適化する。

### 5.1 山登り法による探索
*   **Mutation**: 現在の最強デッキからランダムに1~2枚入れ替える。
*   **Adaptation**: 新しいカードの使い方を数戦学習する（Short-term training）。
*   **Evaluation**: 仮想敵（メタデッキ集団）と対戦し、勝率を測定する。
*   **Selection**: 勝率が向上すれば採用、下がれば棄却。

### 5.2 デッキタイプの自動推論
未知のデッキを渡された際、AIはデッキ内のカードスタッツ平均値（重心）からタイプを推論する。
*   **Aggro**: Early Usage 高, Shield Dmg 高, Min Cost 低
*   **Control**: Hand Adv 高, Board Adv 高, Late Usage 高
*   **Combo**: Cost Discount 高, Hand Var 高

## 6. C++ エンジン実装要件
### 6.1 必須クラス・構造体

```cpp
// カードの動的統計情報を管理
struct CardStats {
    long long play_count;
    double sum_hand_delta;
    double sum_board_delta;
    // ... (16次元分の蓄積変数)
    
    std::vector<float> to_vector(); // 正規化して出力
};

// ゲーム状態管理
class GameState {
public:
    // 特徴量変換 (Pythonへ渡す)
    std::vector<float> vectorize(const CardStats& global_stats);
    
    // 合法手生成 (ルール処理)
    std::vector<Action> get_legal_actions();
    
    // コスト計算 (軽減適用後)
    int get_current_cost(int card_id);
};
```
