# Run artifact manager watcher as a background job (Windows PowerShell)
# Usage: .\scripts\run_artifact_manager.ps1
# This script starts the watcher and writes logs to logs/artifact_manager.log

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$script = Join-Path $projectRoot "training\watch_and_trigger_artifact_manager.py"
$logdir = Join-Path $projectRoot "logs"
if (!(Test-Path $logdir)) { New-Item -ItemType Directory -Path $logdir | Out-Null }
$log = Join-Path $logdir "artifact_manager.log"

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $python
$startInfo.Arguments = "`"$script`" --paths models data --interval 5 --timeout 86400"
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $startInfo
$process.Start() | Out-Null

# async log forwarding
$stdOut = $process.StandardOutput
$stdErr = $process.StandardError

Start-Job -ScriptBlock {
    param($stdOut, $stdErr, $log)
    while (-not $stdOut.EndOfStream) {
        $line = $stdOut.ReadLine()
        $line | Out-File -Append -FilePath $log -Encoding utf8
    }
    while (-not $stdErr.EndOfStream) {
        $line = $stdErr.ReadLine()
        $line | Out-File -Append -FilePath $log -Encoding utf8
    }
} -ArgumentList ($stdOut, $stdErr, $log) | Out-Null

Write-Output "Artifact manager watcher started. Logs: $log"
