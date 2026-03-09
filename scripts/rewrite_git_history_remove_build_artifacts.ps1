[CmdletBinding()]
param(
    [string]$SourceRepoPath = (Get-Location).Path,
    [string]$OutputClonePath,
    [switch]$Force,
    [switch]$SkipGc
)

$ErrorActionPreference = 'Stop'

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Resolve-Path -LiteralPath $Path).Path
}

function Invoke-CheckedGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("git command failed: git " + ($Arguments -join ' '))
    }
}

$sourceRepo = Get-NormalizedPath -Path $SourceRepoPath
$repoName = Split-Path -Path $sourceRepo -Leaf

if (-not $OutputClonePath) {
    $parentDir = Split-Path -Path $sourceRepo -Parent
    $OutputClonePath = Join-Path $parentDir ($repoName + '_history_clean.git')
}

$outputRepo = $OutputClonePath

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'git is required.'
}

if (-not (Get-Command git-filter-repo -ErrorAction SilentlyContinue)) {
    throw 'git-filter-repo is required.'
}

if (Test-Path -LiteralPath $outputRepo) {
    if (-not $Force) {
        throw "Output path already exists: $outputRepo . Use -Force to recreate it."
    }

    Write-Host "[cleanup] Removing existing output clone: $outputRepo"
    Remove-Item -LiteralPath $outputRepo -Recurse -Force
}

# Recurrence prevention: rewrite a disposable clone so local uncommitted work is never destroyed.
$sourceRemoteUrl = ''
Push-Location $sourceRepo
try {
    $sourceRemoteUrl = (git config --get remote.origin.url)
    if ($LASTEXITCODE -ne 0) {
        $sourceRemoteUrl = ''
    }
}
finally {
    Pop-Location
}

Write-Host "[clone] Creating disposable bare mirror at $outputRepo"
Invoke-CheckedGit clone --mirror --no-local "$sourceRepo" "$outputRepo"

Push-Location $outputRepo
try {
    $rewriteArgs = @(
        '--force'
        '--invert-paths'
        '--path', 'build/'
        '--path', 'build-msvc/'
        '--path', 'build-ninja/'
        '--path', 'build_debug/'
        '--path', 'build_vs/'
        '--path', 'build_vs_nmake/'
        '--path', 'build_mingw/'
        '--path-glob', 'build-*/'
        '--path-glob', 'build_*/'
        '--path', 'bin/'
        '--path', 'Debug/'
        '--path', 'Release/'
        '--path', 'RelWithDebInfo/'
        '--path', 'MinSizeRel/'
        '--path', 'CMakeFiles/'
        '--path', 'Testing/'
        '--path', 'CMakeCache.txt'
        '--path', 'cmake_install.cmake'
        '--path', 'compile_commands.json'
        '--path-glob', '*.sln'
        '--path-glob', '*.vcxproj'
        '--path-glob', '*.vcxproj.filters'
        '--path-glob', '*.vcxproj.user'
        '--path-glob', '*.suo'
        '--path-glob', '*.pdb'
        '--path-glob', '*.ilk'
        '--path-glob', '*.lib'
        # Recurrence prevention: drop legacy Windows-invalid paths so filter-repo can complete on Windows hosts.
        '--path', 'c:\temp\game_instance_constructor.txt'
        '--path', 'c:\temp\resolve_debug.txt'
        '--path', 'c:\temp\spell_trigger_debug.txt'
    )

    Write-Host '[rewrite] Removing build artifacts from all history...'
    & git filter-repo @rewriteArgs
    if ($LASTEXITCODE -ne 0) {
        throw 'git filter-repo failed. The mirror clone is preserved for inspection.'
    }

    if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
        Write-Host '[rewrite] Pruning orphaned Git LFS objects...'
        & git lfs prune | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw 'git lfs prune failed.'
        }
    }

    if (-not $SkipGc) {
        Write-Host '[rewrite] Expiring reflogs and compacting rewritten repository...'
        Invoke-CheckedGit reflog expire --expire=now --all
        Invoke-CheckedGit gc --prune=now --aggressive
    }

    Write-Host ''
    Write-Host '[done] History cleanup clone is ready.'
    Write-Host "       Review it at: $outputRepo"
    if ($sourceRemoteUrl) {
        Write-Host "       Original remote was: $sourceRemoteUrl"
    }
    Write-Host '       If the rewritten history is correct, add the remote back and force-push from that mirror clone.'
}
finally {
    Pop-Location
}