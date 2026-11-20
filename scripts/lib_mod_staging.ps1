<#
.SYNOPSIS
    构建或校验 `horizon_physics` 原生 Mod 时生成干净扁平、完全自包含的临时 Staging 工程。
    Horizon 打包规范（horizon-mod-package）仅允许扁平目录结构，拒绝 `crates/` 与 `tests/`
    子路径。本脚本在不破坏开发期独立物理内核 Crate 解耦的前提下，将 `horizon_physics_core`
    源码安全内联进 Staging 工程的 `src/physics_core_embedded/`，以确保产物 `.hmod` 可被玩家
    直接通过 `horizon-launcher native-build` 独立编译通过。
#>

function New-FlatHorizonPhysicsStaging {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModDir,
        [Parameter(Mandatory = $true)]
        [string]$StagingDir
    )

    if (Test-Path $StagingDir) {
        Remove-Item -Path $StagingDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $StagingDir | Out-Null

    # 1. 复制受支持的顶级清单与许可文件
    foreach ($fileName in @("horizon-mod.toml", "Cargo.toml", "LICENSE-MIT", "LICENSE-APACHE")) {
        $srcFile = Join-Path $ModDir $fileName
        if (Test-Path $srcFile) {
            Copy-Item -Path $srcFile -Destination (Join-Path $StagingDir $fileName) -Force
        }
    }

    # 2. 复制资产、数据与 Mod 桥接源码
    foreach ($dirName in @("assets", "data", "src")) {
        $srcSub = Join-Path $ModDir $dirName
        if (Test-Path $srcSub) {
            Copy-Item -Path $srcSub -Destination (Join-Path $StagingDir $dirName) -Recurse -Force
        }
    }

    # 3. 复用已拉取的本地 .horizon/sdk 缓存（使用 robocopy 原生绕过 Windows 260 字符 MAX_PATH 限制）
    $horizonCache = Join-Path $ModDir ".horizon"
    if (Test-Path $horizonCache) {
        $dstCache = Join-Path $StagingDir ".horizon"
        $null = & robocopy $horizonCache $dstCache /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
        if ($LASTEXITCODE -ge 8) {
            throw "复制 .horizon SDK 缓存失败（robocopy exit=$LASTEXITCODE）"
        }
        $global:LASTEXITCODE = 0
    }

    # 4. 将物理内核源码树内联至 src/physics_core_embedded/，满足扁平自包含要求
    $coreSrcDir = Join-Path $ModDir "crates\horizon_physics_core\src"
    if (-not (Test-Path $coreSrcDir)) {
        throw "未找到独立物理内核源码目录: $coreSrcDir"
    }

    $embeddedDir = Join-Path $StagingDir "src\physics_core_embedded"
    New-Item -ItemType Directory -Force -Path $embeddedDir | Out-Null

    $coreModulesPattern = '(?<![\w:])crate::(arena|broadphase|ccd|island|math|narrowphase|solver|voxel|world)\b'
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    foreach ($rsFile in (Get-ChildItem -Path $coreSrcDir -Filter "*.rs" -File)) {
        $targetName = if ($rsFile.Name -eq "lib.rs") { "mod.rs" } else { $rsFile.Name }
        $content = Get-Content -Path $rsFile.FullName -Raw -Encoding UTF8
        $rewritten = [regex]::Replace($content, $coreModulesPattern, 'crate::physics_core_embedded::$1')
        [System.IO.File]::WriteAllText((Join-Path $embeddedDir $targetName), $rewritten, $utf8NoBom)
    }

    # 5. 重写 Staging Cargo.toml：移除本地子 crate 路径与测试依赖
    $stageCargoPath = Join-Path $StagingDir "Cargo.toml"
    $cargoLines = (Get-Content -Path $stageCargoPath -Raw -Encoding UTF8) -split "\r?\n"
    $filteredCargo = New-Object System.Collections.Generic.List[string]
    $skipSection = $false

    foreach ($line in $cargoLines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[dev-dependencies\]$') {
            $skipSection = $true
            continue
        }
        if ($skipSection) {
            if ($trimmed -match '^\[') {
                $skipSection = $false
            } else {
                continue
            }
        }
        if ($trimmed -match '^horizon_physics_core\s*=') {
            continue
        }
        $filteredCargo.Add($line)
    }
    [System.IO.File]::WriteAllText($stageCargoPath, ($filteredCargo -join "`r`n"), $utf8NoBom)

    # 6. 重写 Staging src/*.rs：绑定 crate::horizon_physics_core 模块路径
    $stageLibPath = Join-Path $StagingDir "src\lib.rs"
    $libContent = Get-Content -Path $stageLibPath -Raw -Encoding UTF8
    $libRewritten = [regex]::Replace(
        $libContent,
        '(?m)^pub use horizon_physics_core;\s*$',
        "pub mod physics_core_embedded;`r`npub use physics_core_embedded as horizon_physics_core;"
    )
    [System.IO.File]::WriteAllText($stageLibPath, $libRewritten, $utf8NoBom)

    foreach ($modRs in (Get-ChildItem -Path (Join-Path $StagingDir "src") -Filter "*.rs" -File)) {
        if ($modRs.Name -eq "lib.rs") {
            continue
        }
        $text = Get-Content -Path $modRs.FullName -Raw -Encoding UTF8
        $rewritten = [regex]::Replace(
            $text,
            '(?<![\w:])horizon_physics_core::',
            'crate::horizon_physics_core::'
        )
        [System.IO.File]::WriteAllText($modRs.FullName, $rewritten, $utf8NoBom)
    }

    return $StagingDir
}

function Import-HorizonDevEnv {
    <#
    .SYNOPSIS
        按显式参数 -> HORIZON_DEV_ENV 环境变量 -> 上层工作区默认路径自动加载 MSVC/Horizon 编译环境脚本。
    #>
    [CmdletBinding()]
    param(
        [string]$DevEnvScript = "",
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($DevEnvScript)) {
        $candidates.Add($DevEnvScript)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:HORIZON_DEV_ENV)) {
        $candidates.Add($env:HORIZON_DEV_ENV)
    }
    $parentDir = Split-Path -Parent $RepoRoot
    if (-not [string]::IsNullOrWhiteSpace($parentDir)) {
        $candidates.Add((Join-Path $parentDir "enter-dev-env.ps1"))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            . $candidate
            return $candidate
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($DevEnvScript)) {
        Write-Warning "未找到指定的环境加载脚本 $DevEnvScript，将使用当前会话环境变量继续。"
    } else {
        Write-Host "[Env] 未探测到外置 enter-dev-env.ps1，使用当前会话环境变量继续（可通过 -DevEnvScript 或 HORIZON_DEV_ENV 指定）。" -ForegroundColor DarkGray
    }
    return $null
}

function Resolve-HorizonClientExe {
    <#
    .SYNOPSIS
        按显式参数 -> HORIZON_CLIENT_EXE 环境变量 -> 上层工作区常见构建产物路径解析 horizon-client.exe。
    #>
    [CmdletBinding()]
    param(
        [string]$ClientExe = "",
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($ClientExe)) {
        $candidates.Add($ClientExe)
    }
    if (-not [string]::IsNullOrWhiteSpace($env:HORIZON_CLIENT_EXE)) {
        $candidates.Add($env:HORIZON_CLIENT_EXE)
    }
    $parentDir = Split-Path -Parent $RepoRoot
    if (-not [string]::IsNullOrWhiteSpace($parentDir)) {
        $candidates.Add((Join-Path $parentDir "horizon\target\fast\horizon-client.exe"))
        $candidates.Add((Join-Path $parentDir "horizon\target\release\horizon-client.exe"))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "未找到 horizon-client.exe。请通过 -ClientExe 参数或 `$env:HORIZON_CLIENT_EXE 指定客户端路径。"
}
