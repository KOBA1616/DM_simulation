<#
Clean up generated artifacts in the working tree.

- Safe by default: only removes Python caches unless flags are provided.
- Prompts before making changes unless -Force is specified.

Examples:
  .\scripts\clean_workspace.ps1                 # clean caches (__pycache__ etc)
  .\scripts\clean_workspace.ps1 -CleanBuild     # also remove top-level build/ and bin/
  .\scripts\clean_workspace.ps1 -MoveRootLogs   # move root log/txt outputs into dumps/logs/workspace/
  .\scripts\clean_workspace.ps1 -CleanBuild -MoveRootLogs -Force
#>

param(
    [switch]$Force,
    [switch]$CleanBuild,
    [switch]$CleanCaches,
    [switch]$MoveRootLogs,
    [switch]$IncludeVenvCaches
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not ($CleanBuild -or $CleanCaches -or $MoveRootLogs)) {
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

$candidatesDelete = New-Object 'System.Collections.Generic.List[string]'
$candidatesMove = New-Object 'System.Collections.Generic.List[string]'

# --- Build outputs (top-level only) ---
if ($CleanBuild) {
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "build")
    Add-Candidate $candidatesDelete (Join-Path $projectRoot "bin")
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
}

if ($candidatesDelete.Count -eq 0 -and $candidatesMove.Count -eq 0) {
    Write-Host "Nothing to clean." -ForegroundColor Green
    exit 0
}

Write-Host "Planned actions:" -ForegroundColor Cyan
if ($candidatesDelete.Count -gt 0) {
    Write-Host "\n[Delete]" -ForegroundColor Yellow
    $candidatesDelete | Sort-Object | ForEach-Object { Write-Host " - $_" }
}
if ($candidatesMove.Count -gt 0) {
    Write-Host "\n[Move]" -ForegroundColor Yellow
    $candidatesMove | Sort-Object | ForEach-Object { Write-Host " - $_" }
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
