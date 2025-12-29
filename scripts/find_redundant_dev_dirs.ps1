# Search for possible redundant dev tool directories (mingw, msys, cygwin, llvm-mingw, etc.)
$roots = @(
    'C:\Program Files',
    'C:\Program Files (x86)',
    'C:\Users\ichirou\AppData\Local',
    'C:\Users\ichirou\AppData\Local\Programs',
    'C:\Users\ichirou\Downloads',
    'C:\Users\ichirou'
)
$pattern = '(?i)\b(mingw|msys|cygwin|llvm(-mingw)?|llvm-mingw|mingw64|mingw32|msys2|gcc|g\+\+|clang|gnu|tcc|gdb|make)\b'
$outFile = Join-Path $PSScriptRoot 'redundant_dirs.txt'
if(Test-Path $outFile){ Remove-Item $outFile -Force -ErrorAction SilentlyContinue }
Write-Output "Searching roots and writing results to: $outFile"
$found = [System.Collections.Generic.HashSet[string]]::new()
foreach($r in $roots){
    if(-not (Test-Path $r)) { Write-Output "Root not found: $r"; continue }
    Write-Output "-- Scanning: $r"
    try{
        Get-ChildItem -Path $r -Directory -Recurse -Depth 4 -ErrorAction SilentlyContinue | ForEach-Object {
            if($_.Name -match $pattern -or $_.FullName -match $pattern){
                if($found.Add($_.FullName)){
                    $_.FullName | Tee-Object -FilePath $outFile -Append
                    Write-Output $_.FullName
                }
            }
        }
    } catch {
        Write-Output ('Error scanning {0}: {1}' -f $r, $_.Exception.Message)
    }
}
if($found.Count -eq 0){ Write-Output 'No candidate directories found'; Add-Content $outFile 'No candidate directories found' }
Write-Output "Done. Results in $outFile"