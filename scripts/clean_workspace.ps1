<#
Clean up generated artifacts in the working tree.

- Safe by default: only removes Python caches unless flags are provided.
- Prompts before making changes unless -Force is specified.

Examples:
  .\scripts\clean_workspace.ps1                 # clean caches (__pycache__ etc)
  .\scripts\clean_workspace.ps1 -CleanBuild     # also remove top-level build/ and bin/
  .\scripts\clean_workspace.ps1 -MoveRootLogs   # move root log/txt outputs into dumps/logs/workspace/
  .\scripts\clean_workspace.ps1 -CleanBuild -MoveRootLogs -Force
  .\scripts\clean_workspace.ps1 -PruneModels    # archive old model checkpoints (keep latest 2)
  .\scripts\clean_workspace.ps1 -PruneDepsCache # delete shared .cmake_deps_cache (re-fetched on next build)
#>

param(
    [switch]$Force,
    [switch]$CleanBuild,
    [switch]$CleanCaches,
    [switch]$MoveRootLogs,
    [switch]$MoveRootBinaries,
    [switch]$PruneLogs,
    # 再発防止: 古いモデルチェックポイントが models/ に累積しやすいためビルド時に自動整理する。
    [switch]$PruneModels,
    # 再発防止: FetchContent 共有キャッシュを削除すると次回ビルド時に再取得される。onnxruntime など大容量 DL が発生するため適宜実行すること。
    [switch]$PruneDepsCache,
    [switch]$PruneInactiveBuilds,
    [string]$ActiveBuildDirName = "",
    [int]$KeepLogFiles = 20,
    [int]$LogMaxAgeDays = 14,
    [int]$LogMaxTotalMB = 512,
    [int]$BuildMaxAgeDays = 14,
    [switch]$IncludeVenvCaches,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not ($CleanBuild -or $CleanCaches -or $MoveRootLogs -or $MoveRootBinaries -or $PruneLogs -or $PruneInactiveBuilds -or $PruneModels -or $PruneDepsCache)) {
    # Default behavior: clean caches only.
    $CleanCaches = $true
}

function Add-Candidate([System.Collections.Generic.List[string]]$list, [string]$path) {
    if ([string]::IsNullOrWhiteSpace($path)) { return }
    try {
        if (Test-Path -LiteralPath $path) {
            $full = (Resolve-Path -LiteralPath $path).Path
            if (-not ($list -contains $full)) { $list.Add($full) }
        }
    } catch {
        # ignore
    }
}

function Add-PruneCandidatesByAgeAndCount(
    [System.Collections.Generic.List[string]]$list,
    [string]$baseDir,
    [string[]]$patterns,
    [int]$keepCount,
    [int]$maxAgeDays,
    [Int64]$maxTotalBytes
) {
    if (-not (Test-Path -LiteralPath $baseDir -PathType Container)) { return }

    $matchedFiles = @()
    foreach ($pattern in $patterns) {
        try {
            $matchedFiles += Get-ChildItem -Path $baseDir -File -Recurse -Filter $pattern -ErrorAction SilentlyContinue
        } catch {
            # ignore
        }
    }

    if (-not $matchedFiles) { return }

    $sortedFiles = $matchedFiles |
        Sort-Object @(
            @{ Expression = 'LastWriteTimeUtc'; Descending = $true },
            @{ Expression = 'Length'; Descending = $true }
        )

    if ($keepCount -ge 0 -and $sortedFiles.Count -gt $keepCount) {
        foreach ($file in ($sortedFiles | Select-Object -Skip $keepCount)) {
            Add-Candidate $list $file.FullName
        }
    }

    if ($maxAgeDays -gt 0) {
        $cutoff = (Get-Date).ToUniversalTime().AddDays(-$maxAgeDays)
        foreach ($file in $sortedFiles) {
            if ($file.LastWriteTimeUtc -lt $cutoff) {
                Add-Candidate $list $file.FullName
            }
        }
    }

    if ($maxTotalBytes -gt 0) {
        $remainingFiles = @()
        foreach ($file in $sortedFiles) {
            if (-not ($list -contains $file.FullName)) {
                $remainingFiles += $file
            }
        }

        $remainingTotal = ($remainingFiles | Measure-Object Length -Sum).Sum
        foreach ($file in ($remainingFiles | Sort-Object LastWriteTimeUtc)) {
            if ($remainingTotal -le $maxTotalBytes) {
                break
            }

            Add-Candidate $list $file.FullName
            $remainingTotal -= $file.Length
        }
    }
}

$candidatesDelete = New-Object 'System.Collections.Generic.List[string]'
$candidatesMove = New-Object 'System.Collections.Generic.List[string]'

# --- Build outputs (top-level only) ---
if ($CleanBuild) {
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "build")
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "build-msvc")
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "build-mingw")
    # 再発防止: Ninja のビルド成果物も容量を大きく消費するため、full clean 対象に含める。
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "build-ninja")
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "bin")
}

# --- Inactive build outputs (top-level only) ---
if ($PruneInactiveBuilds) {
    $knownBuildDirs = @("build", "build-msvc", "build-mingw", "build-ninja")
    $cutoff = (Get-Date).ToUniversalTime().AddDays(-$BuildMaxAgeDays)
    foreach ($dirName in $knownBuildDirs) {
        if (-not [string]::IsNullOrWhiteSpace($ActiveBuildDirName) -and $dirName -eq $ActiveBuildDirName) {
            continue
        }

        $path = Join-Path $projectRoot $dirName
        if (-not (Test-Path -LiteralPath $path -PathType Container)) {
            continue
        }

        try {
            $dirInfo = Get-Item -LiteralPath $path -Force
            # 再発防止: 現在使っていない古い build tree だけを対象にし、アクティブ build のキャッシュ破壊を避ける。
            if ($BuildMaxAgeDays -le 0 -or $dirInfo.LastWriteTimeUtc -lt $cutoff) {
                Add-Candidate $candidatesDelete $path
            }
        } catch {
            # ignore
        }
    }
}

# --- Caches (recursive) ---
if ($CleanCaches) {
    $cacheNames = @("__pycache__", ".pytest_cache", ".mypy_cache")
    foreach ($name in $cacheNames) {
        try {
            Get-ChildItem -Path $projectRoot -Recurse -Force -Directory -ErrorAction SilentlyContinue |
                Where-Object {
                    $_.Name -eq $name -and (
                        $IncludeVenvCaches -or (
                            $_.FullName -notlike "*\\.venv\\*" -and
                            $_.FullName -notlike "*\\build\\*" -and
                            $_.FullName -notlike "*\\third_party\\*" -and
                            $_.FullName -notlike "*\\archive\\*"
                        )
                    )
                } |
                ForEach-Object { Add-Candidate $candidatesDelete $_.FullName }
        } catch {
            # ignore
        }
    }
}

# --- Root logs / outputs ---
if ($MoveRootLogs) {
    $rootFiles = @(
        "cmake_configure_mingw.log",
        "crash_diag.log",
        "myeasylog.log",
        "tests_full_run.log",
        "tests_full_run_after_changes.log",
        "test_results.txt",
        "test_results_after_build.txt"
    )
    foreach ($f in $rootFiles) {
        $p = Join-Path $projectRoot $f
        if (Test-Path -LiteralPath $p -PathType Leaf) {
            Add-Candidate $candidatesMove $p
        }
    }

    # Also move transient script logs that sometimes end up under scripts/
    try {
        $scriptsDir = Join-Path $projectRoot "scripts"
        if (Test-Path -LiteralPath $scriptsDir -PathType Container) {
            Get-ChildItem -Path $scriptsDir -File -Filter "cmake_config_*.log" -ErrorAction SilentlyContinue |
                ForEach-Object { Add-Candidate $candidatesMove $_.FullName }
        }
    } catch {
        # ignore
    }
}

# --- Logs (recursive pruning) ---
if ($PruneLogs) {
    # 再発防止: 巨大な debug trace が logs/ に累積しやすいため、件数と経過日数の両方で抑制する。
    Add-PruneCandidatesByAgeAndCount $candidatesDelete (Join-Path $projectRoot "logs") @("*.log", "*.txt", "*.jsonl") $KeepLogFiles $LogMaxAgeDays ($LogMaxTotalMB * 1MB)
    Add-PruneCandidatesByAgeAndCount $candidatesDelete (Join-Path $projectRoot "reports") @("*.log", "*.txt", "*.json") $KeepLogFiles $LogMaxAgeDays ($LogMaxTotalMB * 1MB)
}

# --- Root binary artifacts (opt-in) ---
# These are often produced by local experiments or dependency extraction.
# Moving them can affect runtime DLL discovery on Windows, so keep it explicit.
if ($MoveRootBinaries) {
    $binaryPatterns = @(
        "onnxruntime*.dll",
        "onnxruntime*.pyd"
    )
    foreach ($pat in $binaryPatterns) {
        try {
            Get-ChildItem -Path $projectRoot -File -Filter $pat -ErrorAction SilentlyContinue |
                ForEach-Object { Add-Candidate $candidatesMove $_.FullName }
        } catch {
            # ignore
        }
    }
}

if ($candidatesDelete.Count -eq 0 -and $candidatesMove.Count -eq 0) {
    Write-Host "Nothing to clean." -ForegroundColor Green
    exit 0
}

# --- Old model cleanup (delegate to cleanup_models.py) ---
# 再発防止: models/ に大量の .pth が累積すると数百 MB 消費するため、ビルド前後に自動整理する。
if ($PruneModels) {
    $cleanupModelsScript = Join-Path $scriptDir "cleanup_models.py"
    $pythonExe = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }
    if (Test-Path -LiteralPath $cleanupModelsScript) {
        $modelsDir = Join-Path $projectRoot "models"
        if (Test-Path -LiteralPath $modelsDir) {
            Write-Host "Pruning old model checkpoints..." -ForegroundColor Cyan
            $dryRunFlag = if ($DryRun) { "--dry-run" } else { "--no-dry-run" }
            try {
                & $pythonExe $cleanupModelsScript --models-dir $modelsDir --report $dryRunFlag
            } catch {
                Write-Warning "cleanup_models.py failed: $_"
            }
        }
    } else {
        Write-Warning "cleanup_models.py not found at $cleanupModelsScript"
    }
}

# --- Shared FetchContent deps cache cleanup ---
# 再発防止: .cmake_deps_cache は build-msvc/build-ninja 間で共有するキャッシュ。
# 削除すると次回ビルド時に onnxruntime (~300MB) を再取得するため、容量重視の場合のみ実行する。
if ($PruneDepsCache) {
    $depsCache = Join-Path $projectRoot ".cmake_deps_cache"
    if (Test-Path -LiteralPath $depsCache) {
        Add-Candidate $candidatesDelete $depsCache
    } else {
        Write-Host ".cmake_deps_cache not found (already clean or not yet created)." -ForegroundColor DarkGray
    }
}

if ($candidatesDelete.Count -gt 0) {
    Write-Host "[Delete]" -ForegroundColor Yellow
    $candidatesDelete | Sort-Object | ForEach-Object { Write-Host " - $_" }
}
if ($candidatesMove.Count -gt 0) {
    Write-Host "\n[Move]" -ForegroundColor Yellow
    $candidatesMove | Sort-Object | ForEach-Object { Write-Host " - $_" }
}

if ($DryRun) {
    Write-Host "DryRun enabled. No changes were made." -ForegroundColor DarkYellow
    exit 0
}

if (-not $Force) {
    $confirm = Read-Host "Proceed? Type 'yes' to continue"
    if ($confirm -ne 'yes') {
        Write-Host "Aborted. No changes were made." -ForegroundColor DarkYellow
        exit 0
    }
}

# Execute moves first (keeps logs if deletes include directories)
if ($candidatesMove.Count -gt 0) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $destDir = Join-Path $projectRoot "dumps\logs\workspace\$stamp"
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null

    foreach ($src in ($candidatesMove | Sort-Object)) {
        try {
            $name = Split-Path -Leaf $src
            $dst = Join-Path $destDir $name
            Move-Item -LiteralPath $src -Destination $dst -Force
            Write-Host "Moved: $src -> $dst" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to move: $src ($_ )"
        }
    }
}

# Execute deletes
if ($candidatesDelete.Count -gt 0) {
    foreach ($path in ($candidatesDelete | Sort-Object -Unique)) {
        try {
            if (Test-Path -LiteralPath $path -PathType Container) {
                Remove-Item -LiteralPath $path -Recurse -Force
            } else {
                Remove-Item -LiteralPath $path -Force
            }
            Write-Host "Deleted: $path" -ForegroundColor Green
        } catch {
            Write-Warning "Failed to delete: $path ($_ )"
        }
    }
}

Write-Host "Cleanup complete." -ForegroundColor Green
