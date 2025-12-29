$pattern='(?i)mingw|msys|cygwin|llvm-mingw'
$pathEntries = $env:Path -split ';' | Where-Object {$_ -ne ''}
$matches = $pathEntries | Where-Object {$_ -match $pattern} | Select-Object -Unique
if(-not $matches -or $matches.Count -eq 0){ Write-Output 'NO_MATCHES'; exit 0 }
Write-Output 'MATCHES:'
foreach($p in $matches){ Write-Output $p }

# 削除処理（ユーザーの指示によりバックアップなし）
foreach($p in $matches){
    Write-Output "REMOVING: $p"
    if(Test-Path $p){
        Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Output "RemovedDir: $p"
    } else {
        Write-Output "DirNotFound: $p"
    }
}

# ユーザー PATH を更新
$userPath = [Environment]::GetEnvironmentVariable('Path','User')
if($userPath){
    $new = ($userPath -split ';' | Where-Object { ($_ -ne '') -and -not ($_ -match $pattern) }) -join ';'
    [Environment]::SetEnvironmentVariable('Path',$new,'User')
    Write-Output 'Updated User PATH'
} else {
    Write-Output 'No User PATH'
}

# 現在プロセスの PATH を更新
$env:Path = ($env:Path -split ';' | Where-Object { ($_ -ne '') -and -not ($_ -match $pattern) }) -join ';'
Write-Output 'Current PATH cleaned'
Write-Output 'VERIFY:'
$verify = ($env:Path -split ';') | Where-Object {$_ -match $pattern}
if($verify){ $verify | ForEach-Object { Write-Output "STILL_PRESENT: $_" } } else { Write-Output 'CLEAN' }
