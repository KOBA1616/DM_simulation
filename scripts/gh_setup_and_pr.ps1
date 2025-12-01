<#
gh_setup_and_pr.ps1

Purpose:
- Prompt for (or read) a GitHub Personal Access Token (PAT), set it for gh authentication,
- Optionally add MinGW bin to PATH for this session,
- Push the current branch to origin, create a PR (using gh if available),
- Then call create_pr_and_watch.ps1 to poll workflow runs and download logs if failures.

Usage:
  .\scripts\gh_setup_and_pr.ps1 -Owner 'KOBA1616' -Repo 'DM_simulation'

Notes:
- The script will not echo your token. It will store it in the current session environment variable
  `GITHUB_TOKEN` for subsequent calls. For persistent configuration, use `gh auth login` interactively.
#>

param(
    [Parameter(Mandatory=$true)][string] $Owner,
    [Parameter(Mandatory=$true)][string] $Repo,
    [string] $MinGwPath = 'C:\Program Files (x86)\mingw64\bin'
)

# Helper: Read token securely from stdin if not in environment
if (-not $env:GITHUB_TOKEN) {
    Write-Host "No GITHUB_TOKEN in environment. Please paste a Personal Access Token (scopes: repo, workflow)." -ForegroundColor Yellow
    $secure = Read-Host -AsSecureString "Paste PAT (hidden)"
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $pat = [Runtime.InteropServices.Marshal]::PtrToStringAuto($ptr)
        $env:GITHUB_TOKEN = $pat
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr)
    }
}

# Optionally add MinGW to PATH for this session
if (Test-Path $MinGwPath) {
    if ($env:PATH -notlike "*$MinGwPath*") {
        $env:PATH = $env:PATH + ';' + $MinGwPath
        Write-Host "Added MinGW path to session PATH: $MinGwPath"
    } else {
        Write-Host "MinGW path already in PATH"
    }
} else {
    Write-Host "MinGW path not found at $MinGwPath — skipping PATH update" -ForegroundColor Yellow
}

# Determine current branch and remote
$cwd = Get-Location
Write-Host "Repo root: $cwd"
$branch = git rev-parse --abbrev-ref HEAD
Write-Host "Current branch: $branch"

# Push branch
Write-Host "Pushing branch to origin..."
$push = git push --set-upstream origin $branch 2>&1
Write-Host $push
if ($LASTEXITCODE -ne 0) { Write-Error "git push failed. Fix authentication/remote and retry."; exit 1 }

# If gh CLI is available, login with token and create PR
if (Get-Command gh -ErrorAction SilentlyContinue) {
    Write-Host "gh CLI detected. Logging in using token (non-interactive)..."
    $env:GITHUB_TOKEN | gh auth login --with-token 2>&1 | Write-Host
    Write-Host "Creating PR via gh..."
    $pr = gh pr create --base main --title "test: add card stats bindings test and CI" --body "Adds CI workflow and pytest for card stats bindings (automated)." 2>&1
    Write-Host $pr
    if ($LASTEXITCODE -ne 0) { Write-Error "gh pr create failed" }
    $prUrl = ($pr | Select-String -Pattern 'https://github.com/.+' -AllMatches).Matches.Value | Select-Object -First 1
    if ($prUrl) { Write-Host "PR created: $prUrl" } else { Write-Host "PR created (inspect gh output)." }
} else {
    Write-Host "gh CLI not found. Falling back to REST script create_pr_and_watch.ps1"
    if (-not (Test-Path .\scripts\create_pr_and_watch.ps1)) { Write-Error "create_pr_and_watch.ps1 not found in scripts/"; exit 1 }
    .\scripts\create_pr_and_watch.ps1 -Title "test: add card stats bindings test and CI" -Body "Adds CI workflow and pytest for card stats bindings (automated)." -Head $branch -Owner $Owner -Repo $Repo
}

Write-Host "Done. If workflows were started, check Actions tab or run 'gh run list --workflow ci.yml --branch $branch'."
