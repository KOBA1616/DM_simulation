param(
    [ValidateSet('msvc','mingw')]
    [string]$Toolchain = 'msvc',
    [string]$Config = "Release",
    [string]$Generator = "",
    [switch]$Clean,
    [switch]$EnableCppTests,
    [switch]$UseLibTorch = $false,
    [switch]$UseOnnxRuntime = $false,
    [switch]$SkipAutoCleanup
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent $scriptDir

# Ensure UTF-8 for console output
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

Set-Location $projectRoot

# ----------------------------------------------------------------
# Generator selection: Ninja を優先し，見つからなければ VS17 へフォールバック
# 再発防止: Ninja は -A x64 を受け付けないので Generator 確定後に引数を分岐すること
# ----------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($Generator)) {
    if ($Toolchain -eq 'mingw') {
        $Generator = 'MinGW Makefiles'
    } else {
        # ninja.exe が PATH 上にある場合は Ninja を優先する
        $ninjaAvailable = $null -ne (Get-Command ninja -ErrorAction SilentlyContinue)
        if ($ninjaAvailable) {
            $Generator = 'Ninja'
            Write-Host "Ninja が見つかりました。高速ビルドを使用します (Generator=Ninja)"
        } else {
            $Generator = 'Visual Studio 17 2022'
            Write-Host "Ninja が見つかりませんでした。Visual Studio 17 2022 を使用します"
        }
    }
}

# ビルドディレクトリ: Generator に合わせて分離することで cmake キャッシュの競合を防ぐ
$buildDirName = if ($Toolchain -eq 'mingw') {
    'build-mingw'
} elseif ($Generator -eq 'Ninja') {
    'build-ninja'
} else {
    'build-msvc'
}
$buildDir = Join-Path $projectRoot $buildDirName

if (-not $SkipAutoCleanup) {
    $cleanupScript = Join-Path $scriptDir 'clean_workspace.ps1'
    if (Test-Path -LiteralPath $cleanupScript) {
        Write-Host "Pruning stale build outputs, oversized logs, and old model checkpoints before build..."
        # 再発防止: 現在使う build directory を除外してから古い build/log/models を整理し、容量肥大と誤削除を同時に防ぐ。
        & $cleanupScript -PruneInactiveBuilds -PruneLogs -PruneModels -ActiveBuildDirName $buildDirName -KeepLogFiles 20 -LogMaxAgeDays 14 -LogMaxTotalMB 512 -BuildMaxAgeDays 14 -Force
    }
}

function Invoke-VsDevCmd {
    param(
        [ValidateSet('x64','x86')]
        [string]$Arch = 'x64'
    )

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found at: $vswhere"
    }

    $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installPath)) {
        throw "Visual Studio with MSVC tools not found via vswhere."
    }

    $vsDevCmd = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        # Try fallback to vcvarsall if VsDevCmd isn't present (older VS/BuildTools layouts)
        $vcvarsFallback = Join-Path $installPath 'VC\Auxiliary\Build\vcvarsall.bat'
        if (Test-Path $vcvarsFallback) {
            Write-Host "VsDevCmd.bat not found; falling back to vcvarsall: $vcvarsFallback"
            $vsDevCmd = $vcvarsFallback
        } else {
            throw "VsDevCmd.bat not found at: $vsDevCmd and no vcvarsall fallback found."
        }
    }

    # Import the environment variables produced by VsDevCmd into this PowerShell session.
    # Using cmd.exe because VsDevCmd is a .bat file.
    $cmd = "`"$vsDevCmd`" -no_logo -arch=$Arch -host_arch=$Arch && set"
    cmd.exe /s /c $cmd | ForEach-Object {
        $line = $_
        $idx = $line.IndexOf('=')
        if ($idx -gt 0) {
            $name = $line.Substring(0, $idx)
            $value = $line.Substring($idx + 1)
            try { Set-Item -Path "Env:$name" -Value $value } catch { }
        }
    }
}

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "Cleaning build directory..."
    Remove-Item -Path $buildDir -Recurse -Force
}

if (-not (Test-Path $buildDir)) {
    
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

$cmakeArgs = @("-S", $projectRoot, "-B", $buildDir, "-DCMAKE_BUILD_TYPE=$Config")

# Generator はファイル先頭で確定済み。ここで cmake 引数に追加する
$cmakeArgs += @("-G", $Generator)

if ($Toolchain -eq 'msvc') {
    # cl.exe / MSVC include パスを確保するために VsDevCmd を必ず呼ぶ
    Invoke-VsDevCmd -Arch 'x64'
    if ($Generator -eq 'Ninja') {
        # 再発防止: Ninja は -A x64 を受け付けない。コンパイラは明示指定する。
        $cmakeArgs += @('-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl')
    } else {
        # Visual Studio generators は -A でアーキテクチャを指定する
        $cmakeArgs += @('-A', 'x64')
    }
}

if ($EnableCppTests) {
    Write-Host "Enabling C++ tests: adding -DENABLE_CPP_TESTS=ON to CMake args"
    $cmakeArgs += "-DENABLE_CPP_TESTS=ON"
}

if ($UseLibTorch) {
    Write-Host "Detecting LibTorch path from Python..."
    
    # Try to find python in .venv first
    $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $pythonCmd = $venvPython
    } else {
        $pythonCmd = "python"
    }

    try {
        $torchPath = & $pythonCmd -c "import torch; print(torch.utils.cmake_prefix_path)"
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($torchPath)) {
            Write-Host "Found LibTorch at: $torchPath"
            $cmakeArgs += "-DUSE_LIBTORCH=ON"
            $cmakeArgs += "-DCMAKE_PREFIX_PATH=$torchPath"
        } else {
            Write-Warning "Could not detect LibTorch path via '$pythonCmd'. Ensure 'torch' is installed in the active Python environment."
            $cmakeArgs += "-DUSE_LIBTORCH=ON"
        }
    } catch {
        Write-Warning "Error detecting LibTorch: $_"
        $cmakeArgs += "-DUSE_LIBTORCH=ON"
    }
} else {
    $cmakeArgs += "-DUSE_LIBTORCH=OFF"
}

# 再発防止: 環境にある古い onnxruntime.dll を掴むと native import が API mismatch で失敗し、
# GUI 側で「Outdated C++ Module」誤表示につながるため、既定は ONNX Runtime を無効化する。
# 必要な場合のみ -UseOnnxRuntime を明示して有効化する。
if ($UseOnnxRuntime) {
    $cmakeArgs += "-DUSE_ONNXRUNTIME=ON"
} else {
    $cmakeArgs += "-DUSE_ONNXRUNTIME=OFF"
}

# Googletest integration removed: project no longer fetches or builds googletest.

Write-Host "Configuring (Generator=$Generator, Config=$Config)..."
cmake @cmakeArgs

# 並列ビルド数: 論理コア数を使用してビルドを高速化
$cpuCount = [Environment]::ProcessorCount
Write-Host "Building (parallel=$cpuCount)..."
cmake --build $buildDir --config $Config --parallel $cpuCount
$buildExitCode = $LASTEXITCODE

# 再発防止: Windows + Ninja のリンク時に既存 dm_ai_module.pyd がロックされると
# LNK1104（出力ファイルを開けない）で失敗することがある。
# 失敗時は既存 .pyd を退避してビルドを一度だけ再試行する。
$nativePydPath = Join-Path $projectRoot 'bin\dm_ai_module.cp312-win_amd64.pyd'
$canRetryNativeLink = $IsWindows -and ($Generator -eq 'Ninja') -and (Test-Path -LiteralPath $nativePydPath)
if ($buildExitCode -ne 0 -and $canRetryNativeLink) {
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $backupPath = "$nativePydPath.$timestamp.bak"
    try {
        Move-Item -LiteralPath $nativePydPath -Destination $backupPath -Force
        Write-Warning "Build failed once. Backed up locked native module to: $backupPath"
        Write-Host "Retrying build after native module backup..."
        cmake --build $buildDir --config $Config --parallel $cpuCount
        $buildExitCode = $LASTEXITCODE
    }
    catch {
        Write-Warning "Retry preparation failed while backing up native module: $($_.Exception.Message)"
    }
}

if ($buildExitCode -ne 0) {
    throw "Build failed with exit code $buildExitCode"
}

Write-Host "Build complete. (Generator=$Generator, BuildDir=$buildDirName)"
