<#
Remove common Visual Studio / MSVC generated files and folders from the working tree.
This script prompts before deleting anything. It only removes generated artifacts,
not source files under `src/` or important repository files.
#>

param(
    [switch]$Force
)

Write-Host "Scanning for MSVC/Visual Studio artifacts..."

$patterns = @(
    '*.sln',
    '*.vcxproj',
    '*.vcxproj.filters',
    '*.vcxproj.user',
    'build/',
    'build_*',
    'Debug/',
    'Release/',
    '*.pdb',
    '*.ilk',
    '*.suo'
)

$candidates = @()
foreach ($p in $patterns) {
    $found = Get-ChildItem -Path . -Recurse -Force -ErrorAction SilentlyContinue -Filter $p | Select-Object -Unique -ExpandProperty FullName -ErrorAction SilentlyContinue
    if ($found) { $candidates += $found }
}

if (-not $candidates) {
    Write-Host "No MSVC artifacts found."
    exit 0
}

Write-Host "Found the following artifacts:" -ForegroundColor Yellow
$candidates | ForEach-Object { Write-Host " - $_" }

if (-not $Force) {
    $confirm = Read-Host "Delete these files/directories? Type 'yes' to proceed"
    if ($confirm -ne 'yes') {
        Write-Host "Aborted. No files were deleted."; exit 0
    }
}

foreach ($path in $candidates) {
    try {
        if (Test-Path $path -PathType Container) { Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction Stop }
        else { Remove-Item -LiteralPath $path -Force -ErrorAction Stop }
        Write-Host "Deleted: $path"
    } catch {
        Write-Warning "Failed to remove: $path ($_ )"
    }
}

Write-Host "Cleanup complete." -ForegroundColor Green
