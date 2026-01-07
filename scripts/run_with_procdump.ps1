$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$procdump = Join-Path $repoRoot 'tools\procdump64.exe'
New-Item -ItemType Directory -Force -Path dumps | Out-Null
$proc = Start-Process -FilePath $python -ArgumentList '-m','pytest','-q' -PassThru
Start-Sleep -Milliseconds 200
& $procdump -accepteula -e 1 -ma -x dumps $proc.Id
Wait-Process -Id $proc.Id
Write-Host "procdump finished"
