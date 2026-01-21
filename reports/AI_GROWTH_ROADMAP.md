# AI開発・成長ロードマップ

## 1. 現状の実装分析 (Current Status)

現在、AI関連のコードは目的と依存関係によって大きく2つの領域に分かれています。

### A. 基本学習環境 (`training/`)
- **役割**: 最小限の動作確認とパイプラインのテスト。
- **特徴**: Pythonスタブ（`dm_ai_module.py`）を使用し、C++ビルドなしでも動作します。
- **主要スクリプト**:
  - `simple_game_generator.py`: ルールベース（シールド枚数など）で勝敗を決定し、教師あり学習用データ（`.npz`）を生成。
  - `train_simple.py`: Transformerモデルの基本的な学習ループ。
- **現状の課題**: 生成されるゲーム展開が単純であり、戦略的な深さがありません。

### B. 発展的エコシステム (`dm_toolkit/training/`)
- **役割**: 本格的な強化学習（RL）と進化計算（PBT）。
- **特徴**: C++モジュール（`dm_ai_module`）の高速な並列処理（`ParallelRunner`）と完全なゲームロジックを前提としています。
- **主要スクリプト**:
  - `self_play.py`: AlphaZeroスタイルの自己対戦とデータ収集。
  - `train_pbt.py`: Population Based Training（集団ベースの学習）。デッキやパラメータの異なる複数のエージェントを競わせます。

---

## 2. 今後の方針 (Future Roadmap)

AIを「動く」状態から「強い」状態へ移行させるためのロードマップです。

### フェーズ1: パイプラインの確立（現在）
- **目標**: データの生成から学習、モデル保存までのフローが無停止で完了することを確認する。
- **手段**: `training/` ディレクトリのスクリプトを使用し、エラー（Tensor形状の不一致など）を潰す。

### フェーズ2: C++統合と自己対戦
- **目標**: 本格的なルールに基づく自己対戦の実現。
- **手段**:
  1. `scripts/build.ps1` でC++拡張をビルド。
  2. `dm_toolkit/training/self_play.py` を有効化し、ランダムではなく「過去の自分」と戦うループを構築。

### フェーズ3: メタゲームの進化
- **目標**: プレイングだけでなく、デッキ構築の最適化。
- **手段**: `train_pbt.py` を稼働させ、勝率の高いデッキ構成を進化的に探索させる。

---

## 3. 具体的な実行手順 (Actionable Steps)

### 手順1: ベースラインモデルの作成
まず、Pythonのみで完結する環境で学習パイプラインを回します。

```bash
# 1. データの生成
python training/simple_game_generator.py
# -> data/simple_training_data.npz が生成されます

# 2. モデルの学習
python training/train_simple.py --data-path data/simple_training_data.npz --epochs 5
# -> models/duel_transformer_YYYYMMDD.pth が保存されます
```

### 手順2: 自己対戦による強化学習 (要C++ビルド)
C++モジュールが利用可能な場合、より高度な学習を行います。

```bash
# 自己対戦の実行（例）
python dm_toolkit/training/self_play.py
```

### 手順3: 集団学習 (PBT) の実行
複数のエージェントを用いた進化計算を実行します。

```bash
python dm_toolkit/training/train_pbt.py --gens 10 --pop 4
```

## 4. 必要な準備
- **C++環境**: 本格的な学習には `bin/dm_ai_module.pyd` (Windows) または `.so` (Linux) が必要です。
- **設定ファイル**: `data/cards.json` や `data/meta_decks.json` が最新であることを確認してください。
