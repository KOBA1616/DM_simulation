$python = "C:\Users\ichirou\DM_simulation\.venv\Scripts\python.exe"
$procdump = "C:\Users\ichirou\DM_simulation\tools\procdump64.exe"
New-Item -ItemType Directory -Force -Path dumps | Out-Null
$proc = Start-Process -FilePath $python -ArgumentList '-m','pytest','-q' -PassThru
Start-Sleep -Milliseconds 200
& $procdump -accepteula -e 1 -ma -x dumps $proc.Id
Wait-Process -Id $proc.Id
Write-Host "procdump finished"
