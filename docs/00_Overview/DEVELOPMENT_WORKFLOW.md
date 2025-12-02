# Development Workflow & Environment Setup

This document outlines the standard development workflow using GitHub CLI (`gh`) and the environment setup for the DM_simulation project.

## 1. Environment Setup

Before starting development, ensure your environment variables are correctly configured.

### Windows (PowerShell)
Run the initialization script to add necessary tools (like `gh`) to your PATH for the current session.

```powershell
. .\scripts\init_env.ps1
```

This script checks for:
- GitHub CLI (`gh`)
- CMake
- Python
- Git

## 2. GitHub CLI (`gh`) Workflow

We use `gh` for managing Pull Requests, CI monitoring, and releases.

### Authentication
Ensure you are logged in:
```powershell
gh auth login
gh auth status
```

### CI Monitoring
To check the status of GitHub Actions workflows:

```powershell
# List recent runs
gh run list

# View details of a specific run (replace <run-id>)
gh run view <run-id>

# View logs of a specific run
gh run view <run-id> --log
```

### Pull Requests
Create and manage PRs directly from the terminal:

```powershell
# Create a PR from the current branch
gh pr create --title "feat: description" --body "Details..."

# Check PR status
gh pr status

# Merge a PR (when checks pass)
gh pr merge <pr-number> --merge --delete-branch
```

## 3. CI/CD Pipeline

The project uses GitHub Actions for Continuous Integration.
- **Workflow File**: `.github/workflows/windows-build.yml`
- **Triggers**: Push to `main`, Pull Requests.

### Troubleshooting CI Failures
1. Use `gh run list` to identify the failing run.
2. Download logs using `gh run view <id> --log > build.log`.
3. Analyze the log for compilation or test errors.
4. Reproduce locally if possible (using `scripts/setup_mingw_env.ps1` or Visual Studio).

## 4. Directory Structure Standards

- `scripts/`: Helper scripts for setup and maintenance.
- `docs/`: Project documentation.
- `src/`: C++ source code.
- `python/`: Python bindings and scripts.
