$run = Join-Path $PSScriptRoot '..\scripts\run_gui.ps1'
try {
    Write-Host "Invoking $run -AllowFallback"
    & $run -AllowFallback
} catch {
    Write-Host "CATCH START"
    $_ | Format-List * -Force
    if ($_.InvocationInfo) {
        Write-Host "ScriptName: $($_.InvocationInfo.ScriptName)"
        Write-Host "Line: $($_.InvocationInfo.ScriptLineNumber)"
        Write-Host "Position: $($_.InvocationInfo.OffsetInLine)"
        Write-Host "LineText: $($_.InvocationInfo.Line)"
    }
    Write-Host "CATCH END"
    exit 1
}
