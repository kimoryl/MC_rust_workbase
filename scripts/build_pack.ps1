<#
.SYNOPSIS
    构建并校验 MC_rust_workbase 物理内核（horizon_physics_core）、全部自研 Mod（workbase_core、
    horizon_physics 等 .hmod）以及整合包（.hpack），支持自动回填 SHA-256 清单并输出汇总报告。
.EXAMPLE
    .\scripts\build_pack.ps1
    .\scripts\build_pack.ps1 -UpdateManifest
    .\scripts\build_pack.ps1 -SkipModpackCheck
#>
[CmdletBinding()]
param(
    [switch]$UpdateManifest,
    [switch]$SkipCore,
    [switch]$SkipPack,
    [switch]$SkipModpackCheck,
    [string]$DevEnvScript = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$DistDir = Join-Path $RepoRoot "dist"
$ModsDir = Join-Path $RepoRoot "mods"
$PackDir = Join-Path $RepoRoot "pack"
$CoreCrateDir = Join-Path $ModsDir "horizon_physics\crates\horizon_physics_core"
$ManifestPath = Join-Path $PackDir "horizon-modpack.toml"
. (Join-Path $PSScriptRoot "lib_mod_staging.ps1")

function Write-Section {
    param([string]$Title)
    Write-Host "`n============================================================" -ForegroundColor DarkCyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor DarkCyan
}

function Get-TomlStringValue {
    param(
        [string]$Content,
        [string]$Key,
        [string]$DefaultValue = ""
    )
    if ($Content -match "(?m)^\s*$([regex]::Escape($Key))\s*=\s*`"([^`"]+)`"") {
        return $matches[1]
    }
    return $DefaultValue
}

function Update-ModpackManifestMods {
    param(
        [string]$Path,
        [System.Collections.Generic.List[object]]$ModArtifacts
    )

    if (-not (Test-Path $Path) -or $ModArtifacts.Count -eq 0) {
        return
    }

    $original = Get-Content -Path $Path -Raw -Encoding UTF8
    # 移除既有生效的 [[mods]] 区块（保留顶部元信息与注释说明），重新以确定性顺序写入
    $lines = $original -split "\r?\n"
    $kept = New-Object System.Collections.Generic.List[string]
    $inModsBlock = $false

    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -eq "[[mods]]") {
            $inModsBlock = $true
            continue
        }
        if ($inModsBlock) {
            if ($trimmed -match '^\[' -and $trimmed -ne "[[mods]]") {
                $inModsBlock = $false
                $kept.Add($line)
            }
            continue
        }
        $kept.Add($line)
    }

    while ($kept.Count -gt 0 -and [string]::IsNullOrWhiteSpace($kept[$kept.Count - 1])) {
        $kept.RemoveAt($kept.Count - 1)
    }

    $kept.Add("")
    foreach ($art in $ModArtifacts) {
        $kept.Add("[[mods]]")
        $kept.Add("id = `"$($art.Id)`"")
        $kept.Add("version = `"$($art.Version)`"")
        $kept.Add("sha256 = `"$($art.Sha256)`"")
        $kept.Add("on = true")
        $kept.Add("")
    }

    $updated = ($kept -join "`r`n")
    [System.IO.File]::WriteAllText($Path, $updated, (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "[Manifest] 已将 $($ModArtifacts.Count) 项 .hmod SHA-256 哈希回填至 $Path" -ForegroundColor Green
}

# 0. 加载 MSVC 2022 编译环境
Write-Section "0. 加载编译环境"
$null = Import-HorizonDevEnv -DevEnvScript $DevEnvScript -RepoRoot $RepoRoot

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
$summary = New-Object System.Collections.Generic.List[object]
$modArtifacts = New-Object System.Collections.Generic.List[object]

# 1. 编译与 Release 测试 horizon_physics_core
if (-not $SkipCore) {
    Write-Section "1. 编译并测试物理内核 horizon_physics_core (--release)"
    $coreCargo = Join-Path $CoreCrateDir "Cargo.toml"
    if (Test-Path $coreCargo) {
        Push-Location $CoreCrateDir
        try {
            Write-Host "[Cargo] cargo test --release" -ForegroundColor Yellow
            cargo test --release
            if ($LASTEXITCODE -ne 0) {
                throw "horizon_physics_core release 测试失败（exit=$LASTEXITCODE）"
            }

            Write-Host "[Cargo] cargo build --release" -ForegroundColor Yellow
            cargo build --release
            if ($LASTEXITCODE -ne 0) {
                throw "horizon_physics_core release 构建失败（exit=$LASTEXITCODE）"
            }

            $summary.Add([PSCustomObject]@{
                Type    = "crate"
                Id      = "horizon_physics_core"
                Version = (Get-TomlStringValue -Content (Get-Content $coreCargo -Raw) -Key "version" -DefaultValue "0.1.0")
                File    = "horizon_physics_core (release lib)"
                Sha256  = "-"
                Status  = "OK (test+build --release)"
            })
        } finally {
            Pop-Location
        }
    } else {
        Write-Warning "未发现 $coreCargo，跳过物理内核构建。"
    }
}

# 2. 按确定性顺序构建并校验全部自研 Mod（优先 workbase_core 与 horizon_physics）
Write-Section "2. 构建并校验自研 Mod (.hmod)"
$preferredOrder = @("workbase_core", "horizon_physics")
$discoveredMods = @(Get-ChildItem -Path $ModsDir -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path (Join-Path $_.FullName "horizon-mod.toml") })

$orderedMods = New-Object System.Collections.Generic.List[System.IO.DirectoryInfo]
foreach ($name in $preferredOrder) {
    $matched = $discoveredMods | Where-Object { $_.Name -eq $name }
    if ($matched) { $orderedMods.Add($matched) }
}
foreach ($d in ($discoveredMods | Sort-Object Name)) {
    if ($preferredOrder -notcontains $d.Name) {
        $orderedMods.Add($d)
    }
}

foreach ($mod in $orderedMods) {
    Write-Host "`n[Build Mod] $($mod.Name)" -ForegroundColor Cyan
    $modTomlPath = Join-Path $mod.FullName "horizon-mod.toml"
    $modTomlText = Get-Content -Path $modTomlPath -Raw -Encoding UTF8
    $modId = Get-TomlStringValue -Content $modTomlText -Key "id" -DefaultValue $mod.Name
    $modVer = Get-TomlStringValue -Content $modTomlText -Key "version" -DefaultValue "0.1.0"

    $needsFlatStage = Test-Path (Join-Path $mod.FullName "crates\horizon_physics_core\Cargo.toml")
    $buildRoot = $mod.FullName
    $stageDir = $null
    $prevTargetDir = $env:CARGO_TARGET_DIR
    $locationPushed = $false

    try {
        if ($needsFlatStage) {
            $stageDir = Join-Path $env:TEMP ("hzs" + $PID)
            Write-Host "[Stage] 生成自包含扁平包目录: $stageDir" -ForegroundColor DarkCyan
            $buildRoot = New-FlatHorizonPhysicsStaging -ModDir $mod.FullName -StagingDir $stageDir
            $env:CARGO_TARGET_DIR = Join-Path $env:TEMP "hzpt"
        }

        Push-Location $buildRoot
        $locationPushed = $true
        $isWasm = $false
        $cargoTomlPath = Join-Path $buildRoot "Cargo.toml"
        if (Test-Path $cargoTomlPath) {
            $cargoText = Get-Content -Path $cargoTomlPath -Raw -Encoding UTF8
            if ($cargoText -match '\[package\.metadata\.horizon\]') {
                $isWasm = $true
            }
        }

        if ($needsFlatStage) {
            Write-Host "[Verify Self-Contained Build] cargo check --lib & cargo horizon-mod check" -ForegroundColor Yellow
            cargo check --lib
            if ($LASTEXITCODE -ne 0) {
                throw "自包含 Mod 编译验证失败: $($mod.Name)（exit=$LASTEXITCODE）"
            }
            cargo horizon-mod check
            if ($LASTEXITCODE -ne 0) {
                throw "Horizon Mod 静态表面检查失败: $($mod.Name)（exit=$LASTEXITCODE）"
            }
        }

        if ($isWasm) {
            cargo horizon-mod build --wasm
        } else {
            cargo horizon-mod build
        }
        if ($LASTEXITCODE -ne 0) {
            throw "构建 Mod 失败: $($mod.Name)（exit=$LASTEXITCODE）"
        }

        $outDir = Join-Path $buildRoot "target\horizon-mod"
        $builtFiles = @(Get-ChildItem -Path $outDir -Filter "*.hmod" -ErrorAction SilentlyContinue)
        if ($builtFiles.Count -eq 0) {
            throw "未在 $outDir 找到构建生成的 .hmod 文件（$($mod.Name)）"
        }

        foreach ($f in $builtFiles) {
            $dest = Join-Path $DistDir $f.Name
            Copy-Item -Path $f.FullName -Destination $dest -Force
            Write-Host "[Check Mod] $($f.Name)" -ForegroundColor Green
            cargo horizon-mod check $dest
            if ($LASTEXITCODE -ne 0) {
                throw "cargo horizon-mod check 失败: $($f.Name)（exit=$LASTEXITCODE）"
            }

            $hash = (Get-FileHash -Algorithm SHA256 -Path $dest).Hash.ToLowerInvariant()
            Write-Host "  ID     : $modId ($modVer)"
            Write-Host "  SHA256 : $hash"

            $artifact = [PSCustomObject]@{
                Type    = "hmod"
                Id      = $modId
                Version = $modVer
                File    = $f.Name
                Sha256  = $hash
                Status  = "OK (checked)"
            }
            $modArtifacts.Add($artifact)
            $summary.Add($artifact)
        }
    } finally {
        if ($locationPushed) {
            Pop-Location
        }
        if ($null -eq $prevTargetDir) {
            Remove-Item Env:CARGO_TARGET_DIR -ErrorAction SilentlyContinue
        } else {
            $env:CARGO_TARGET_DIR = $prevTargetDir
        }
        if ($stageDir -and (Test-Path $stageDir)) {
            Remove-Item -Path $stageDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# 3. 回填清单（可选）并打包校验 .hpack 整合包
if (-not $SkipPack -and (Test-Path $ManifestPath)) {
    Write-Section "3. 打包并校验整合包 (.hpack)"
    if ($UpdateManifest) {
        Update-ModpackManifestMods -Path $ManifestPath -ModArtifacts $modArtifacts
    }

    $manifestText = Get-Content -Path $ManifestPath -Raw -Encoding UTF8
    $packId = Get-TomlStringValue -Content $manifestText -Key "id" -DefaultValue "mc_rust_workbase"
    $packVer = Get-TomlStringValue -Content $manifestText -Key "version" -DefaultValue "0.1.0"
    $hpackName = "$packId-$packVer.hpack"
    $hpackPath = Join-Path $DistDir $hpackName

    if (Test-Path $hpackPath) {
        Remove-Item -Path $hpackPath -Force
    }

    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::Open($hpackPath, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $ManifestPath, "horizon-modpack.toml"
        ) | Out-Null

        $overridesDir = Join-Path $PackDir "overrides"
        if (Test-Path $overridesDir) {
            $overrideFiles = @(Get-ChildItem -Path $overridesDir -Recurse -File |
                Where-Object { $_.Name -ne ".gitkeep" })
            foreach ($file in $overrideFiles) {
                $rel = $file.FullName.Substring($PackDir.Length + 1).Replace('\', '/')
                [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $file.FullName, $rel) | Out-Null
            }
        }
    } finally {
        $zip.Dispose()
    }

    $packStatus = "PACKED"
    if (-not $SkipModpackCheck) {
        Write-Host "[Check Pack] $hpackName" -ForegroundColor Green
        cargo horizon-mod check $hpackPath
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "整合包校验未通过；若清单尚未包含生效的 [[mods]]，请尝试添加 -UpdateManifest 参数。"
            exit $LASTEXITCODE
        }
        $packStatus = "OK (checked)"
    } else {
        $packStatus = "PACKED (check skipped)"
    }

    $packHash = (Get-FileHash -Algorithm SHA256 -Path $hpackPath).Hash.ToLowerInvariant()
    $summary.Add([PSCustomObject]@{
        Type    = "hpack"
        Id      = $packId
        Version = $packVer
        File    = $hpackName
        Sha256  = $packHash
        Status  = $packStatus
    })
}

# 4. 输出汇总清单
Write-Section "4. 构建产物汇总清单"
$summary | Format-Table -Property Type, Id, Version, File, Status, Sha256 -AutoSize
Write-Host "[Done] 输出目录: $DistDir" -ForegroundColor Green
