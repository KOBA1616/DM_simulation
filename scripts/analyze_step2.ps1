# Step 2: TurnStats設計調査スクリプト

Write-Host "`n=== Step 2: TurnStats設計の現状調査 ===" -ForegroundColor Cyan

Write-Host "`n現在のTurnStats構造:" -ForegroundColor Yellow
Write-Host @"
struct TurnStats {
    int played_without_mana = 0;          // プレイヤーごと？ターンごと？
    int cards_drawn_this_turn = 0;        // プレイヤーごと？ターンごと？
    int cards_discarded_this_turn = 0;    // プレイヤーごと？ターンごと？
    int creatures_played_this_turn = 0;   // プレイヤーごと？ターンごと？
    int spells_cast_this_turn = 0;        // プレイヤーごと？ターンごと？
    int current_chain_depth = 0;          // ゲーム全体？
    bool mana_charged_by_player[2];       // ✅ プレイヤーごと（修正済み）
};
"@ -ForegroundColor White

Write-Host "`n使用箇所の検索中..." -ForegroundColor Yellow

$fields = @(
    "played_without_mana",
    "cards_drawn_this_turn",
    "cards_discarded_this_turn",
    "creatures_played_this_turn",
    "spells_cast_this_turn",
    "current_chain_depth"
)

foreach ($field in $fields) {
    Write-Host "`n[$field]" -ForegroundColor Cyan
    $matches = Select-String -Pattern "turn_stats\.$field" -Path src\**\*.cpp, src\**\*.hpp -CaseSensitive
    
    if ($matches.Count -gt 0) {
        Write-Host "  使用箇所: $($matches.Count)件" -ForegroundColor White
        $matches | Select-Object -First 3 | ForEach-Object {
            $file = Split-Path $_.Path -Leaf
            Write-Host "    - $file : Line $($_.LineNumber)" -ForegroundColor Gray
        }
        if ($matches.Count -gt 3) {
            Write-Host "    ... 他 $($matches.Count - 3)件" -ForegroundColor Gray
        }
    } else {
        Write-Host "  使用箇所なし" -ForegroundColor Gray
    }
}

Write-Host "`n`n=== 検討事項 ===" -ForegroundColor Yellow
Write-Host @"
1. これらのフィールドはプレイヤーごとに管理すべきか？
   
   DMルールの観点:
   - cards_drawn_this_turn: 通常はアクティブプレイヤーのみ（ターンごとでOK）
   - creatures_played_this_turn: カード効果で参照される場合あり（プレイヤーごと？）
   - spells_cast_this_turn: 同上
   - current_chain_depth: 効果の連鎖深度（ゲーム全体）

2. 設計オプション:
   
   Option A: 現状維持（ターンごと）
     - 実装変更なし
     - アクティブプレイヤーの統計のみ追跡
   
   Option B: 完全プレイヤーごと化
     struct TurnStats {
         int played_without_mana[2];
         int cards_drawn_this_turn[2];
         // ...全フィールドを配列化
     };
   
   Option C: プレイヤー統計を分離
     struct PlayerTurnStats {
         int cards_drawn = 0;
         int creatures_played = 0;
         // ...
     };
     PlayerTurnStats per_player_stats[2];
     TurnStats global_stats;  // chain_depthなど

3. 推奨アプローチ:
   - まずは Option A (現状維持)
   - 実際に問題が起きたら Option C に移行
   - カード効果実装時に要件が明確になる

"@ -ForegroundColor White

Write-Host "`n次のアクション:" -ForegroundColor Yellow
Write-Host "  - Step 1.5完了後、しばらくは現状維持で様子見" -ForegroundColor White
Write-Host "  - Step 3 (PhaseManager移行) を優先" -ForegroundColor White
Write-Host "  - カード効果実装時に再検討" -ForegroundColor White
Write-Host ""
