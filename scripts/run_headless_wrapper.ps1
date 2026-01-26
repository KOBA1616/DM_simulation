$ErrorActionPreference = 'Stop'
$projectRoot = 'c:\Users\ichirou\DM_simulation'
$env:PYTHONPATH = $projectRoot
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$log = Join-Path $projectRoot "logs\headless_run_output_wrapper.txt"
& $pythonExe "$projectRoot\scripts\headless_mainwindow.py" --auto 10 *> $log
exit $LASTEXITCODE
