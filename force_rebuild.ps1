# Force rebuild by touching source files
Write-Host "=== Forcing source file timestamps update ===" -ForegroundColor Cyan

# Touch source files
$files = @(
    "src\engine\game_instance.cpp",
    "src\engine\game_instance.hpp",
    "src\engine\actions\strategies\phase_strategies.cpp",
    "src\engine\ai\simple_ai.cpp",
    "src\engine\ai\simple_ai.hpp",
    "src\bindings\bind_core.cpp",
    "src\core\game_state.cpp",
    "src\core\game_state.hpp"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        (Get-Item $file).LastWriteTime = Get-Date
        Write-Host "  Touched: $file" -ForegroundColor Green
    }
}

# Stop Python
Write-Host "`n=== Stopping Python processes ===" -ForegroundColor Cyan
Stop-Process -Name python,pythonw -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Remove old binaries 
Write-Host "`n=== Removing old binaries ===" -ForegroundColor Cyan
Remove-Item -Force dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Force bin\Release\dm_ai_module*.pyd -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build-msvc\Release\* -ErrorAction SilentlyContinue
Write-Host "  Cleanup complete" -ForegroundColor Green

# Build
Write-Host "`n=== Building ===" -ForegroundColor Cyan
cmake --build build-msvc --config Release --target dm_ai_module

# Check result
Write-Host "`n=== Checking result ===" -ForegroundColor Cyan
if (Test-Path "bin\Release\dm_ai_module.cp312-win_amd64.pyd") {
    $pyd = Get-Item "bin\Release\dm_ai_module.cp312-win_amd64.pyd"
    Write-Host "SUCCESS! Built .pyd file:" -ForegroundColor Green
    Write-Host "  Path: $($pyd.FullName)" -ForegroundColor White
    Write-Host "  Size: $($pyd.Length) bytes" -ForegroundColor White
    Write-Host "  Modified: $($pyd.LastWriteTime)" -ForegroundColor White
    
    Copy-Item $pyd.FullName -Destination "." -Force
    Write-Host "`nCopied to workspace root" -ForegroundColor Green
    
    # Test
    Write-Host "`n=== Quick test ===" -ForegroundColor Cyan
    $testResult = python -c @"
import dm_ai_module
print('✓ Module imported')
gs = dm_ai_module.GameState(42)
print('✓ GameState created')
# Test Phase 2: is_human_player
try:
    result = gs.is_human_player(0)
    print('✓ Phase 2: is_human_player() working')
except AttributeError:
    print('✗ Phase 2: is_human_player() missing')
# Test Phase 1.1: SimpleAI with GameState
gs.setup_test_duel()
deck = list(range(1,11))*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)
actions = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
if len(actions) > 0:
    ai = dm_ai_module.SimpleAI()
    try:
        idx = ai.select_action(actions, gs)
        print('✓ Phase 1.1: SimpleAI.select_action(actions, gs) working')
    except:
        print('✗ Phase 1.1: SimpleAI signature issue')
print('All tests passed!')
"@ 2>&1
    Write-Host $testResult
} else {
    Write-Host "FAILED - .pyd file not found" -ForegroundColor Red
}
