# Run migration canary: enable telemetry and run targeted tests

# Usage: Open PowerShell in repo root and run:
#    .\scripts\run_migration_canary.ps1

$env:DM_MIGRATION_TELEMETRY = "1"
$env:DM_MIGRATION_METRICS_PATH = "$(Resolve-Path .)\migration_metrics_canary.jsonl"

Write-Host "DM_MIGRATION_TELEMETRY=$env:DM_MIGRATION_TELEMETRY"
Write-Host "Metrics path: $env:DM_MIGRATION_METRICS_PATH"

# Run a focused set of tests that exercise conversion and load/save behavior
pytest -q tests/unit/test_load_lift_migration.py tests/unit/test_action_converter_advanced.py tests/unit/test_action_converter_mekraid.py

Write-Host "Canary run complete. Collected metrics at: $env:DM_MIGRATION_METRICS_PATH"
