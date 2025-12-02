# シナリオ・トレーニング（詰将棋モード）実装設計書
## ～ 特定盤面からのコンボ・ループ習得システム ～

## 1. 概要
本機能は、ゲームを「初期状態（0ターン目）」からではなく、意図的に操作された「特定の盤面」から開始させるデバッグおよび学習機能である。

### 目的
- **コンボ習得**: 特定のカードが揃った状態からスタートし、無限ループや即死コンボの手順をAIに過学習させる。
- **リーサル訓練**: 相手のシールドが残りわずかで、自分の手札が潤沢な状態など、詰めろの局面を反復練習させる。
- **バグ特定**: 特定の状況下でのみ発生するエンジンの挙動を再現・修正する。

## 2. C++ エンジン実装 (Game Engine)
ゲームの状態（GameState）を外部から強制的に上書きするインターフェースを実装する。

### 2.1 シナリオ定義構造体
シナリオを構成する要素を定義する。JSON等で記述し、C++でパースすることを想定。

```cpp
struct ScenarioConfig {
    // 自分のリソース
    int my_mana = 0;
    std::vector<int> my_hand_cards;    // 手札のカードIDリスト
    std::vector<int> my_battle_zone;   // 場のクリーチャーIDリスト
    std::vector<int> my_mana_zone;     // マナゾーンのカードIDリスト
    std::vector<int> my_grave_yard;    // 墓地のカードIDリスト
    
    // 相手のリソース (棒立ちの練習台、あるいは妨害役)
    int enemy_shield_count = 5;
    std::vector<int> enemy_battle_zone;  // 妨害クリーチャー（メタクリ）など
    bool enemy_can_use_trigger = false;  // トリガー使用の可否
    
    // 勝利条件のオーバーライド (オプション)
    // 通常の勝利に加え、「ループ証明完了」などを条件にする場合
    bool loop_proof_mode = false;        // 特定のアクションをN回繰り返したら勝ちとする
};
```

### 2.2 エンジン拡張 (reset_with_scenario)
通常の `reset()` とは別に、シナリオ設定を受け取る初期化関数を追加する。

```cpp
class GameInstance {
public:
    // 通常の初期化
    void reset(int deck_id_1, int deck_id_2);

    // シナリオモード初期化
    void reset_with_scenario(const ScenarioConfig& config) {
        // 1. メモリのクリア
        clear_all_zones();

        // 2. プレイヤーの状態設定
        state.current_turn = 5; // コンボができそうなターン数に設定
        state.active_player = PLAYER_1;

        // 3. 指定カードの配置
        // IDからCardオブジェクトを生成し、各ゾーンのvectorに直接pushする
        for (int id : config.my_hand_cards) {
            Card* c = card_factory.create(id);
            state.players[PLAYER_1].hand.push_back(c);
        }

        for (int id : config.my_battle_zone) {
            Card* c = card_factory.create(id);
            // 召喚酔いフラグなどを解除しておく
            c->is_summoning_sickness = false;
            state.players[PLAYER_1].battle_zone.push_back(c);
        }

        // マナの設定 (指定がなければアンタップ状態で配置)
        state.players[PLAYER_1].mana = config.my_mana;
        setup_mana_zone(config.my_mana_zone);

        // 4. 相手の状態設定
        setup_enemy(config);

        // 5. 状態キャッシュの更新 (Legal Actionsの再計算)
        state.update_cache();
    }
};
```

### 2.3 ループ検知・証明ロジック (Loop Detector)
無限ループの練習をさせる場合、AIが実際に無限回操作するのは不可能なため、「ループに入ったこと」を検知する仕組みが必要。

```cpp
// 簡易的なループ検知
// 「全く同じ盤面」が「同一ターン内に」「3回」出現したらループ成立とみなす
bool check_infinite_loop(const GameState& current_state) {
    size_t state_hash = current_state.calculate_hash();

    // ハッシュの出現回数をカウント
    state.history[state_hash]++;

    if (state.history[state_hash] >= 3) {
        // ループ証明完了（勝利扱い）
        return true;
    }
    return false;
}
```

## 3. Python 学習環境実装 (Training)
強化学習ループの初期段階（または定期的な特訓フェーズ）で、このシナリオモードを呼び出す。

### 3.1 シナリオデータの定義
Python辞書形式で、習得させたいコンボパターンを定義する。

```python
SCENARIOS = {
    "infinite_draw_loop": {
        "my_hand": [101, 102],  # コンボパーツA, B
        "my_mana": 5,
        "my_battle_zone": [103], # コンボ始動役
        "enemy_battle_zone": [],
        "description": "AとBと場のアタックトリガーで無限ドロー"
    },
    "lethal_puzzle_1": {
        "my_hand": [201, 202, 203], # スピードアタッカー等
        "my_mana": 6,
        "enemy_shield": 2,
        "enemy_blocker": [301], # ブロッカー1体
        "description": "ブロッカーを除去しつつリーサルを取る"
    }
}
```

### 3.2 特訓用学習ループ (The Drill)
通常の対戦とは異なる報酬設計で学習させる。

```python
def run_scenario_training(agent, scenario_name, episodes=1000):
    config = SCENARIOS[scenario_name]

    for _ in range(episodes):
        # C++エンジンをシナリオモードでリセット
        # (Pythonラッパー経由で C++の reset_with_scenario を呼ぶ)
        state = env.reset(scenario=config)

        done = False
        steps = 0
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # --- 報酬シェイピング (ここが重要) ---
            # シナリオ特有の「正解行動」にボーナスを与える

            # 例: ループ証明成功
            if info.get("loop_proven"):
                reward += 10.0
            
            # 例: コンボパーツを間違ってマナに埋めたら罰則
            if action.type == "MANA_CHARGE" and action.card_id in config["my_hand"]:
                reward -= 5.0
                done = True # 即終了してやり直し

            # ステップ数罰則（最短手順を目指させるため）
            reward -= 0.1

            agent.update(state, action, reward, next_state)
            state = next_state
            steps += 1
            
            if steps > 50: # 手順が長すぎたら失敗
                break
```

### 3.3 データ作成支援ツール (GUI Tool Support)
シナリオ定義（JSON）を手動で記述するのはコストが高いため、将来的には以下のフローを推奨する。
- **Generic Card Generator (GUI)**:
    - 盤面エディタ機能を追加し、GUI上でカードを配置して状態を作成。
    - "Export Scenario JSON" ボタンで `SCENARIOS` 形式のテキストを出力。
    - これにより、複雑な盤面も直感的に作成可能とする。

## 4. 統合と運用フロー
1.  **コンボ発掘**: 人間が「このカードとこのカードでループできそう」と思ったら、その状況を `SCENARIOS` に定義する（将来的にはGUIツールを活用）。
2.  **集中特訓**: Kaggle Notebookの最初の30分を使って `run_scenario_training` を実行し、AIに手順を丸暗記させる。
3.  **実戦投入**: その後、通常の対戦学習（Self-Play）に切り替える。
    *   AIは「あの状況（シナリオに近い盤面）になれば勝てる」と知っているため、実戦でもその盤面を目指す（パーツを集める、マナを貯める）ようになる。

## 5. まとめ
*   **C++**: 盤面を強制セットアップする `reset_with_scenario` と、同一局面検知による `check_infinite_loop` を実装する。
*   **Python**: コンボパーツを定義し、それを無駄遣いすると罰則を与える特訓ループを回す。
*   **効果**: 確率の低いランダム探索に頼らず、複雑なループコンボを確実かつ短時間でAIに伝授できる。
