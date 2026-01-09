# Phase 4 実装開始前の確認事項と逆質問

**作成日**: 2026年1月9日  
# Phase 4 決定事項（逆質問の確定版）

**確定日**: 2026年1月9日  
**目的**: 逆質問で決定した内容のみを簡潔に保持し、実装に直結させる

---

## ✅ 決定サマリー（Q1-Q9）

| 質問 | 決定 | 実装時期 | メモ |
|------|------|---------|------|
| Q1: Synergy初期化 | A 手動定義 | Week 2 Day 1 | JSON + `from_manual_pairs()` |
| Q2: CLSトークン | A 先頭 | Week 2 Day 1 | `[CLS] [GLOBAL] [SEP] ...` |
| Q3: バッチサイズ | 8→16→32→64 | Week 2 Day 1-3 | 推奨 32 |
| Q4: データ生成 | A 新規作成 | Week 2 Day 1 | `generate_transformer_training_data.py` |
| Q5: Positional Encoding | A 学習可能 | Week 2 Day 1 | `nn.Parameter` で学習 |
| Q6: データ拡張 | カスタム方式 | Week 2 Day 1 | Deck正規化 + Battle重なり保持 |
| Q7: 評価指標 | あると便利まで | Week 2-3 | vs Random, vs MLP, ターン数, 速度, Entropy |
| Q8: デプロイ基準 | バランス型 (B) | Week 3 | vs MLP ≥55% & <10ms, 24h推奨 |
| Q9: Synergy Matrix | A 密行列 | Week 2 Day 1 | 4MB、最適化後回し |

---

## 🔧 Q6 データ拡張の実装要点
- Deck/Hand/Mana/Grave: 順序無関係 → ソートで正規化
- Battle: 重なり(スタック)は保持、それ以外は正規化
- 空ゾーンは省略しない（セパレータを必ず含める）
- ドロップアウト: Phase 2 まで実施しない

---

## 📅 実装ロードマップ（抜粋）
- Week 2 Day 1: Synergy JSON、データ生成1000件、pos_embedding学習可、正規化前処理、Synergy密行列
- Week 2 Day 2-3: 指標実装（vs Random/MLP、ターン数、速度）、バッチ拡大、Loss可視化
- Week 3: vs MLP ≥55%、推論<10ms、24hテスト、Go/No-Go

---

## 参照
- 決定詳細: PHASE4_DECISIONS_FINAL.md
- 状況サマリー: docs/00_Overview/05_Transformer_Current_Status.md
- Day1詳細計画: docs/00_Overview/06_Week2_Day1_Detailed_Plan.md
- マスター要件: docs/00_Overview/00_Status_and_Requirements_Summary.md
- [ ] `generate_transformer_training_data.py` で 1000 サンプル生成
- [ ] DuelTransformer: 学習可能 pos_embedding
- [ ] データ前処理: Deck正規化 + Battle保持
- [ ] Synergy 密行列 初期化

### Week 2 Day 2-3（1月14-16日） - 訓練 & 評価
- [ ] 評価指標: vs Random, vs MLP, 平均ターン数, 推論時間
- [ ] バッチサイズ拡大テスト (8→16→32)
- [ ] Loss 曲線の可視化

### Week 3（1月20-24日） - 最適化 & デプロイ準備
- [ ] vs MLP ≥ 55% 達成確認
- [ ] 推論速度 < 10ms 実測
- [ ] 24時間連続稼働テスト
- [ ] デプロイ Go/No-Go 判定

---

**確定日**: 2026年1月9日  
**ユーザー確認**: ✅  
**実装責任**: AI Code Generation Agent  
**次レビュー**: Week 2 Day 2
