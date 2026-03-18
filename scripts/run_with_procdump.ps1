param(
	[string]$PythonExe,
	[string[]]$PytestArgs = @('-m', 'pytest', '-q'),
	[string]$DumpDir = 'dumps',
	[string]$ProcDumpExe,
	[switch]$CollectEventLog,
	[switch]$DryRun
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path

if (-not $PythonExe) {
	$PythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
}
if (-not $ProcDumpExe) {
	$ProcDumpExe = Join-Path $repoRoot 'tools\procdump64.exe'
}

$dumpRoot = if ([System.IO.Path]::IsPathRooted($DumpDir)) { $DumpDir } else { Join-Path $repoRoot $DumpDir }
New-Item -ItemType Directory -Force -Path $dumpRoot | Out-Null

Write-Host "[run_with_procdump] Python: $PythonExe"
Write-Host "[run_with_procdump] ProcDump: $ProcDumpExe"
Write-Host "[run_with_procdump] DumpDir: $dumpRoot"
Write-Host "[run_with_procdump] Target args: $($PytestArgs -join ' ')"

if ($DryRun) {
	Write-Host "[run_with_procdump] DryRun enabled. No process will be started."
	return
}

if (-not (Test-Path $PythonExe)) {
	throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $ProcDumpExe)) {
	throw "ProcDump executable not found: $ProcDumpExe"
}

$startTime = Get-Date
$proc = Start-Process -FilePath $PythonExe -ArgumentList $PytestArgs -PassThru
Start-Sleep -Milliseconds 300

# 再発防止: 異常終了時の再現情報不足を避けるため、First-Chance ではなく
# unhandled exception (-e 1) を full dump (-ma) で採取する。
& $ProcDumpExe -accepteula -e 1 -ma -x $dumpRoot $proc.Id
$procdumpExit = $LASTEXITCODE

Wait-Process -Id $proc.Id
$targetExit = $proc.ExitCode
$endTime = Get-Date

if ($CollectEventLog) {
	$eventPath = Join-Path $dumpRoot 'windows_application_errors.txt'
	try {
		# 再発防止: 0xC0000409 / 3221226505 を含む Windows 側クラッシュ痕跡を
		# 収集し、dump と同じフォルダに保存して調査手順を単純化する。
		Get-WinEvent -FilterHashtable @{ LogName = 'Application'; StartTime = $startTime.AddMinutes(-1) } |
			Where-Object {
				$_.Id -in 1000, 1001 -or
				$_.Message -match '0xC0000409|3221226505|dm_ai_module|python\.exe|pytest'
			} |
			Select-Object TimeCreated, Id, ProviderName, LevelDisplayName, Message |
			Format-List |
			Out-File -FilePath $eventPath -Encoding utf8
		Write-Host "[run_with_procdump] Event log exported: $eventPath"
	}
	catch {
		Write-Warning "[run_with_procdump] Failed to export event log: $($_.Exception.Message)"
	}
}

$metaPath = Join-Path $dumpRoot 'run_with_procdump_meta.txt'
@(
	"start_time=$($startTime.ToString('o'))"
	"end_time=$($endTime.ToString('o'))"
	"python=$PythonExe"
	"procdump=$ProcDumpExe"
	"args=$($PytestArgs -join ' ')"
	"target_exit_code=$targetExit"
	"procdump_exit_code=$procdumpExit"
) | Out-File -FilePath $metaPath -Encoding utf8

Write-Host "[run_with_procdump] Target exit code: $targetExit"
Write-Host "[run_with_procdump] ProcDump exit code: $procdumpExit"
Write-Host "[run_with_procdump] Metadata: $metaPath"

# pytest 側の終了コードを返す（CI 判定を維持）
exit $targetExit
