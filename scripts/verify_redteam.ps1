<#
.SYNOPSIS
    MC_rust_workbase 自动化红队合规与全链路门禁脚本。
    依次执行：
      1) 根级严格静态合规测试（STRICT_FULL_AUDIT=1 + cargo test --test redteam_compliance）
      2) horizon_physics_core 全量单元测试与 Release 模式基准测试
      3) mods/horizon_physics FakeHost 集成测试与 cargo horizon-mod check
      4) 代码格式化与 Clippy 静态检查（当本地安装 rustfmt / clippy 时执行）
    并输出结构化门禁汇总报告。
.EXAMPLE
    .\scripts\verify_redteam.ps1
    .\scripts\verify_redteam.ps1 -JsonReport .\dist\redteam-report.json
#>
[CmdletBinding()]
param(
    [string]$DevEnvScript = "",
    [string]$JsonReport = "",
    [switch]$SkipOptionalLint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CoreCrateDir = Join-Path $RepoRoot "mods\horizon_physics\crates\horizon_physics_core"
$PhysicsModDir = Join-Path $RepoRoot "mods\horizon_physics"
. (Join-Path $PSScriptRoot "lib_mod_staging.ps1")
$Gates = New-Object System.Collections.Generic.List[object]
$HasFailure = $false

function Invoke-Gate {
    param(
        [string]$GateName,
        [string]$Target,
        [string]$WorkingDir,
        [scriptblock]$Action
    )

    Write-Host "`n[GATE] $GateName ($Target)" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $status = "PASS"
    $detail = "exit=0"

    Push-Location $WorkingDir
    try {
        & $Action
        if ($LASTEXITCODE -ne 0) {
            $status = "FAIL"
            $detail = "exit=$LASTEXITCODE"
            $script:HasFailure = $true
        }
    } catch {
        $status = "FAIL"
        $detail = $_.Exception.Message
        $script:HasFailure = $true
    } finally {
        Pop-Location
        $sw.Stop()
    }

    $color = if ($status -eq "PASS") { "Green" } elseif ($status -eq "SKIP") { "Yellow" } else { "Red" }
    Write-Host "  -> [$status] ($($sw.Elapsed.TotalSeconds.ToString('F2'))s) $detail" -ForegroundColor $color

    $script:Gates.Add([PSCustomObject]@{
        Gate        = $GateName
        Target      = $Target
        Status      = $status
        DurationSec = [math]::Round($sw.Elapsed.TotalSeconds, 2)
        Detail      = $detail
    })
}

function Add-SkippedGate {
    param(
        [string]$GateName,
        [string]$Target,
        [string]$Reason
    )
    Write-Host "`n[GATE] $GateName ($Target)" -ForegroundColor DarkYellow
    Write-Host "  -> [SKIP] $Reason" -ForegroundColor Yellow
    $script:Gates.Add([PSCustomObject]@{
        Gate        = $GateName
        Target      = $Target
        Status      = "SKIP"
        DurationSec = 0.0
        Detail      = $Reason
    })
}

# 0. 加载编译环境
$null = Import-HorizonDevEnv -DevEnvScript $DevEnvScript -RepoRoot $RepoRoot

$prevStrict = $env:STRICT_FULL_AUDIT
$env:STRICT_FULL_AUDIT = "1"

try {
    # 1. 根级严格静态合规门禁
    Invoke-Gate -GateName "01. Root Static Compliance (STRICT_FULL_AUDIT=1)" `
        -Target "tests/redteam_compliance.rs" `
        -WorkingDir $RepoRoot `
        -Action {
            cargo test --test redteam_compliance -- --nocapture
        }

    # 2. 物理内核全量单元测试
    if (Test-Path (Join-Path $CoreCrateDir "Cargo.toml")) {
        Invoke-Gate -GateName "02. Core Engine Unit Tests" `
            -Target "horizon_physics_core" `
            -WorkingDir $CoreCrateDir `
            -Action {
                cargo test --all-targets
            }

        # 3. 物理内核 Release 测试与基准测试
        Invoke-Gate -GateName "03. Core Engine Release & Benchmarks" `
            -Target "horizon_physics_core (--release)" `
            -WorkingDir $CoreCrateDir `
            -Action {
                cargo test --release
                if ($LASTEXITCODE -ne 0) { return }
                $benchesDir = Join-Path $CoreCrateDir "benches"
                if (Test-Path $benchesDir) {
                    $benchFiles = @(Get-ChildItem -Path $benchesDir -Filter "*.rs" -ErrorAction SilentlyContinue)
                    if ($benchFiles.Count -gt 0) {
                        cargo bench
                    }
                }
            }
    } else {
        $HasFailure = $true
        $Gates.Add([PSCustomObject]@{
            Gate        = "02. Core Engine Unit Tests"
            Target      = "horizon_physics_core"
            Status      = "FAIL"
            DurationSec = 0.0
            Detail      = "missing $CoreCrateDir\Cargo.toml"
        })
    }

    # 4. 原生桥接 Mod FakeHost 集成测试与 cargo horizon-mod check
    if (Test-Path (Join-Path $PhysicsModDir "Cargo.toml")) {
        Invoke-Gate -GateName "04. Mod Bridge Integration Tests (FakeHost)" `
            -Target "mods/horizon_physics" `
            -WorkingDir $PhysicsModDir `
            -Action {
                cargo test
            }

        Invoke-Gate -GateName "05. Horizon Mod Static Check & Staging Build" `
            -Target "mods/horizon_physics (.hmod)" `
            -WorkingDir $PhysicsModDir `
            -Action {
                $stageDir = Join-Path $env:TEMP ("hzr" + $PID)
                $prevTargetDir = $env:CARGO_TARGET_DIR
                try {
                    $null = New-FlatHorizonPhysicsStaging -ModDir $PhysicsModDir -StagingDir $stageDir
                    $env:CARGO_TARGET_DIR = Join-Path $env:TEMP "hzpt"
                    Push-Location $stageDir
                    try {
                        cargo check --lib
                        if ($LASTEXITCODE -ne 0) { return }
                        cargo horizon-mod check
                        if ($LASTEXITCODE -ne 0) { return }
                        cargo horizon-mod build
                        if ($LASTEXITCODE -ne 0) { return }
                        cargo horizon-mod check "target\horizon-mod\horizon_physics.hmod"
                    } finally {
                        Pop-Location
                    }
                } finally {
                    if ($null -eq $prevTargetDir) {
                        Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
                    } else {
                        $env:CARGO_TARGET_DIR = $prevTargetDir
                    }
                    if (Test-Path $stageDir) {
                        Remove-Item -Path $stageDir -Recurse -Force -ErrorAction SilentlyContinue
                    }
                }
            }
    } else {
        $HasFailure = $true
        $Gates.Add([PSCustomObject]@{
            Gate        = "04. Mod Bridge Integration Tests (FakeHost)"
            Target      = "mods/horizon_physics"
            Status      = "FAIL"
            DurationSec = 0.0
            Detail      = "missing $PhysicsModDir\Cargo.toml"
        })
    }

    # 5. 可选静态格式化与 Clippy 门禁（全仓三包统一校验）
    if (-not $SkipOptionalLint) {
        $hasFmt = $false
        $hasClippy = $false
        try {
            $null = cargo fmt --version 2>$null
            if ($LASTEXITCODE -eq 0) { $hasFmt = $true }
        } catch { $hasFmt = $false }
        try {
            $null = cargo clippy --version 2>$null
            if ($LASTEXITCODE -eq 0) { $hasClippy = $true }
        } catch { $hasClippy = $false }

        $lintTargets = @(
            @{ Name = "Root Compliance"; Dir = $RepoRoot },
            @{ Name = "horizon_physics_core"; Dir = $CoreCrateDir },
            @{ Name = "horizon_physics"; Dir = $PhysicsModDir }
        )

        if ($hasFmt) {
            Invoke-Gate -GateName "06. Rustfmt Check (All Packages)" `
                -Target "root + core + mod_bridge" `
                -WorkingDir $RepoRoot `
                -Action {
                    foreach ($item in $lintTargets) {
                        Push-Location $item.Dir
                        try {
                            cargo fmt --check
                            if ($LASTEXITCODE -ne 0) { return }
                        } finally {
                            Pop-Location
                        }
                    }
                }
        } else {
            Add-SkippedGate -GateName "06. Rustfmt Check (All Packages)" `
                -Target "root + core + mod_bridge" `
                -Reason "rustfmt component not installed"
        }

        if ($hasClippy) {
            Invoke-Gate -GateName "07. Clippy Strict Check (All Packages)" `
                -Target "root + core + mod_bridge" `
                -WorkingDir $RepoRoot `
                -Action {
                    foreach ($item in $lintTargets) {
                        Push-Location $item.Dir
                        try {
                            cargo clippy --all-targets -- -D warnings
                            if ($LASTEXITCODE -ne 0) { return }
                        } finally {
                            Pop-Location
                        }
                    }
                }
        } else {
            Add-SkippedGate -GateName "07. Clippy Strict Check (All Packages)" `
                -Target "mc_rust_workbase_compliance" `
                -Reason "clippy component not installed"
        }
    }
} finally {
    if ($null -eq $prevStrict) {
        Remove-Item Env:STRICT_FULL_AUDIT -ErrorAction SilentlyContinue
    } else {
        $env:STRICT_FULL_AUDIT = $prevStrict
    }
}

# 输出结构化门禁报告
Write-Host "`n============================================================" -ForegroundColor DarkCyan
Write-Host "  MC_rust_workbase Red-Team Gate Report" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor DarkCyan
$Gates | Format-Table -Property Gate, Target, Status, DurationSec, Detail -AutoSize

$passedCount = @($Gates | Where-Object { $_.Status -eq "PASS" }).Count
$failedCount = @($Gates | Where-Object { $_.Status -eq "FAIL" }).Count
$skippedCount = @($Gates | Where-Object { $_.Status -eq "SKIP" }).Count
$overall = if ($HasFailure) { "FAILED" } else { "PASSED" }

Write-Host ("Summary: {0} | PASS={1} FAIL={2} SKIP={3}" -f $overall, $passedCount, $failedCount, $skippedCount) `
    -ForegroundColor $(if ($HasFailure) { "Red" } else { "Green" })

if (-not [string]::IsNullOrWhiteSpace($JsonReport)) {
    $reportDir = Split-Path -Parent $JsonReport
    if (-not [string]::IsNullOrWhiteSpace($reportDir)) {
        New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
    }
    $payload = [PSCustomObject]@{
        timestamp_utc = (Get-Date).ToUniversalTime().ToString("o")
        overall       = $overall
        passed        = $passedCount
        failed        = $failedCount
        skipped       = $skippedCount
        gates         = $Gates
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -Path $JsonReport -Encoding UTF8
    Write-Host "JSON 报告已写入: $JsonReport" -ForegroundColor Cyan
}

if ($HasFailure) {
    exit 1
}
exit 0
