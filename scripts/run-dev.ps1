<#
.SYNOPSIS
    使用本机源码构建的 horizon-client.exe 快速启动调试指定自研 Mod。
.EXAMPLE
    .\scripts\run-dev.ps1 -ModName workbase_core
#>
param(
    [string]$ModName = "workbase_core",
    [string]$ClientExe = "",
    [string]$DevEnvScript = "",
    [string]$ResourcesDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ModDir = Join-Path $RepoRoot "mods\$ModName"
. (Join-Path $PSScriptRoot "lib_mod_staging.ps1")

if (-not (Test-Path $ModDir)) {
    throw "未找到 Mod 目录: $ModDir"
}
$resolvedClient = Resolve-HorizonClientExe -ClientExe $ClientExe -RepoRoot $RepoRoot
$null = Import-HorizonDevEnv -DevEnvScript $DevEnvScript -RepoRoot $RepoRoot

Push-Location $ModDir
try {
    if ($ResourcesDir -ne "") {
        cargo horizon-mod run --game $resolvedClient -- --resources $ResourcesDir
    } else {
        cargo horizon-mod run --game $resolvedClient
    }
} finally {
    Pop-Location
}
