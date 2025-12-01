<#
PowerShell helper: create_pr_and_watch.ps1

Purpose:
- Create a PR from the current branch to `main` using the GitHub REST API.
- Poll workflow runs for that PR and stream their status; download and show logs if failed.

Requirements:
- Set environment variable GITHUB_TOKEN with a Personal Access Token that has `repo` scope.
- Run in repo root where `git` is available and branch is pushed to origin.

Usage:
  $env:GITHUB_TOKEN = '<YOUR_TOKEN>'
  .\scripts\create_pr_and_watch.ps1 -Title 'CI: run Windows build & tests' -Body 'Add Windows workflow and tests' -Head 'ci/windows-tests' -Owner 'KOBA1616' -Repo 'DM_simulation'

#>

param(
    [Parameter(Mandatory=$true)] [string] $Title,
    [Parameter(Mandatory=$true)] [string] $Body,
    [Parameter(Mandatory=$true)] [string] $Head,
    [Parameter(Mandatory=$true)] [string] $Owner,
    [Parameter(Mandatory=$true)] [string] $Repo
)

if (-not $env:GITHUB_TOKEN) {
    Write-Error "Environment variable GITHUB_TOKEN is not set. Create a PAT and set it: `$env:GITHUB_TOKEN = '<TOKEN>'"
    exit 1
}

$headers = @{ Authorization = "token $env:GITHUB_TOKEN"; Accept = 'application/vnd.github+json'; 'User-Agent' = 'powershell' }

$prBody = @{ title = $Title; head = $Head; base = 'main'; body = $Body }

Write-Host "Creating PR $Head -> main on $Owner/$Repo..."
$resp = Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$Owner/$Repo/pulls" -Headers $headers -Body ($prBody | ConvertTo-Json -Depth 6)
if ($null -eq $resp) { Write-Error "Failed to create PR"; exit 2 }

$prUrl = $resp.html_url
Write-Host "PR created: $prUrl"

Write-Host 'Polling workflow runs for PR...' -ForegroundColor Cyan

function Get-WorkflowRunsForRef($ref) {
    $uri = "https://api.github.com/repos/$Owner/$Repo/actions/runs?event=pull_request&per_page=50"
    $runs = Invoke-RestMethod -Method Get -Uri $uri -Headers $headers
    return $runs.workflow_runs | Where-Object { $_.head_branch -eq $Head }
}

for ($i=0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 5
    $runs = Get-WorkflowRunsForRef -ref $Head
    if (-not $runs) { Write-Host "No workflow runs found yet... (poll $i)"; continue }
    foreach ($r in $runs) {
        Write-Host "Run: $($r.name) id=$($r.id) status=$($r.status) conclusion=$($r.conclusion) workflow=$($r.workflow_id)" -ForegroundColor Green
    }
    $active = $runs | Where-Object { $_.status -eq 'in_progress' -or $_.status -eq 'queued' }
    if ($active) { continue }
    break
}

$runs = Get-WorkflowRunsForRef -ref $Head
if (-not $runs) { Write-Host 'No workflow runs found. Check Actions tab on GitHub.'; exit 0 }

$failed = $runs | Where-Object { $_.conclusion -ne 'success' }
if ($failed) {
    Write-Host "Some workflow runs failed or were canceled. Will download logs for the first failing run." -ForegroundColor Yellow
    $first = $failed[0]
    $logsUri = "https://api.github.com/repos/$Owner/$Repo/actions/runs/$($first.id)/logs"
    $outZip = "workflow_logs_$($first.id).zip"
    Write-Host "Downloading logs for run id=$($first.id) to $outZip..."
    Invoke-RestMethod -Method Get -Uri $logsUri -Headers $headers -OutFile $outZip
    Write-Host "Downloaded logs to $outZip. Please inspect locally or share the file if you need help." -ForegroundColor Cyan
    exit 3
} else {
    Write-Host 'All workflow runs succeeded.' -ForegroundColor Green
    exit 0
}
