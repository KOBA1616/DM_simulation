$ErrorActionPreference = 'Stop'
$projectRoot = 'c:\Users\ichirou\DM_simulation'
$env:PYTHONPATH = $projectRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$input = Join-Path $projectRoot "data\test_cards.json"
$output = Join-Path $projectRoot "data\test_cards_migrated.json"
& $pythonExe "$projectRoot\scripts\migrate_actions_to_commands.py" $input $output
exit $LASTEXITCODE
