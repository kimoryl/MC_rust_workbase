<#
.SYNOPSIS
    兼容入口：转发调用 scripts/build_pack.ps1 执行全链路构建与校验。
#>
[CmdletBinding()]
param(
    [switch]$UpdateManifest,
    [switch]$SkipCore,
    [switch]$SkipPack,
    [switch]$SkipModpackCheck
)

$targetScript = Join-Path $PSScriptRoot "build_pack.ps1"
& $targetScript `
    -UpdateManifest:$UpdateManifest `
    -SkipCore:$SkipCore `
    -SkipPack:$SkipPack `
    -SkipModpackCheck:$SkipModpackCheck
exit $LASTEXITCODE
