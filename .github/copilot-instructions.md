# デュエル・マスターズ シミュレーションプロジェクト

## プロジェクト概要
AIを搭載したデュエル・マスターズTCGシミュレーター（C++/Pythonハイブリッド構成）

## アーキテクチャ
- **C++コア** (`src/`): ゲームエンジン、MCTS AI、高速処理
- **Pythonツールキット** (`dm_toolkit/`): GUI、学習、ユーティリティ
- **ビルドシステム**: CMake（Ninja推奨）
- **モジュール**: `dm_ai_module` - pybind11でC++をPythonバインド
- **移行方針**: 現在は機能のC++移行を優先して進めています。移行中は互換性確保と自動テストを重視し、Pythonフォールバックは暫定的なサポートとしてください。

## コーディング規約
- エラー修正時，"必ず"再発防止のためのコメントを編集ファイルの追加してください．
### Python
- 全ての関数・変数に型ヒントを付ける（mypy準拠）
- `dm_ai_module.pyi`からスタブ型をインポート
- ネイティブ(C++)とフォールバック(Python)の両実装に対応
- `DM_DISABLE_NATIVE=1`でPython実装を強制可能

### C++
- C++17標準に準拠
- 共有所有は`std::shared_ptr`、排他所有は`std::unique_ptr`
- 読み取り専用パラメータは`const`参照
- pybind11バインディングは`src/bindings/`に配置

### テスト
- `tests/`配下にpytestで記述、ファイル名は`test_*.py`
- C++/Python連携の統合テストを含める

## ゲームロジック

### カードゾーン
- **Deck/Hand/Mana Zone/Battle Zone/Graveyard/Shield Zone**

### ターンフェーズ
1. **StartPhase**: アンタップ、状態リセット 
2. **DrawPhase**: 1枚ドロー（先攻1ターン目スキップ）
3. **MainPhase**: マナチャージ、召喚、呪文詠唱
4. **AttackPhase**: アンタップクリーチャーで攻撃宣言
5. **EndPhase**: クリーンアップ

### 重要概念
- **召喚酔い**: 召喚ターンは攻撃不可（スピードアタッカー除く）`turns_in_play`で管理
- **マナコスト**: 文明マナの一致が必要
- **ブロッカー**: 攻撃を防げるクリーチャー

## 再発防止チェックリスト

### 🚨 頻出バグと対策
1. **ゾーン整合性エラー**
   - ❌ PythonまたはC++片方のみゾーン更新 → 状態不一致
   - ✅ 両方のゾーン状態を同期更新、テストで検証

2. **プレイヤーID間違い**
   - ❌ 1始まりで実装 → インデックスエラー
   - ✅ 0始まり（Player 0, Player 1）厳守

3. **カードID重複**
   - ❌ カード種別IDをインスタンスIDとして使用
   - ✅ カードインスタンスごとに一意なIDを生成

4. **フェーズ遷移イベント漏れ**
   - ❌ フェーズ変更時にイベント未発火 → AI判断ミス
   - ✅ 必ずフェーズ変更イベントをトリガー

5. **召喚酔い未チェック**
   - ❌ `turns_in_play`を確認せず攻撃許可
   - ✅ 攻撃前に`turns_in_play >= 1`またはスピードアタッカー判定

6. **マナ計算ミス**
   - ❌ 総マナ量のみチェック → 文明不一致で詠唱失敗
   - ✅ 総量と文明マナの両方を検証

## よく使うコマンド

```powershell
# ビルド
.\scripts\rebuild_clean.ps1      # クリーンビルド
.\scripts\quick_build.ps1        # インクリメンタルビルド

# 実行・テスト
.\scripts\run_gui.ps1            # GUI起動
pytest tests/ -v                 # 全テスト実行
mypy dm_toolkit/                 # 型チェック

# デバッグ
python debug_*.py                # 個別コンポーネントテスト
python test_deck_placement.py   # デッキ配置検証
python test_multi_turn.py       # マルチターンシミュレーション
```

## 環境変数
- `DM_DISABLE_NATIVE=1`: Python実装を強制
- `PYTHONUTF8=1`, `PYTHONPATH=.`: UTF-8とパス設定

## ファイル命名規則
- Python: `snake_case.py` / C++: `snake_case.cpp, .hpp`
- テスト: `test_<feature>.py` / スクリプト: `動詞_名詞.ps1`

## 機能実装時の注意
- 既存設計ドキュメント（`PHASE_AWARE_AI_DESIGN.md`等）を事前確認
- C++とPython両方の実装を一貫性を保って更新
- フォールバック動作を考慮（C++拡張がない環境）
- ゲーム状態のシリアライズ/デシリアライズを検証
