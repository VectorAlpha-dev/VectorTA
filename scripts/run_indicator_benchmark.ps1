param(
    [string]$TargetDir = "",
    [string]$Out = "benchmarks\\indicator_benchmark_results.json",
    [string]$Match = "",
    [string]$CargoFeatures = "",
    [switch]$AllFeatures,
    [string]$RUSTFLAGS = "",
    [string]$Toolchain = "",
    [switch]$OnlyBatch,
    [string]$BatchSizes = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    if ($env:CARGO_TARGET_DIR) {
        $TargetDir = $env:CARGO_TARGET_DIR
    } else {
        $TargetDir = Join-Path $RepoRoot "target"
    }
}

$TargetDir = [System.IO.Path]::GetFullPath($TargetDir)

Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:SCCACHE_DISABLE = "1"
$env:CARGO_TARGET_DIR = $TargetDir

if ($OnlyBatch) {
    $env:INDICATOR_BENCH_ONLY_BATCH = "1"
} else {
    Remove-Item Env:INDICATOR_BENCH_ONLY_BATCH -ErrorAction SilentlyContinue
}

if ([string]::IsNullOrWhiteSpace($BatchSizes)) {
    Remove-Item Env:INDICATOR_BENCH_BATCH_SIZES -ErrorAction SilentlyContinue
} else {
    $env:INDICATOR_BENCH_BATCH_SIZES = $BatchSizes

    if ([string]::IsNullOrWhiteSpace($Match) -and $OnlyBatch) {
        $singleTok = @(
            $BatchSizes.Split(@(',', ' ', "`t", "`r", "`n"), [System.StringSplitOptions]::RemoveEmptyEntries)
        )
        if ($singleTok.Count -eq 1) {
            $pretty = switch ($singleTok[0].Trim().ToLowerInvariant()) {
                { $_ -in @("10k", "10000") } { "10k"; break }
                { $_ -in @("100k", "100000") } { "100k"; break }
                { $_ -in @("1m", "1000000") } { "1M"; break }
                default { "" }
            }
            if (-not [string]::IsNullOrWhiteSpace($pretty)) {
                $Match = "batch/$pretty"
            }
        }
    }
}

if ([string]::IsNullOrWhiteSpace($RUSTFLAGS)) {
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
} else {
    $env:RUSTFLAGS = $RUSTFLAGS
}

$criterionDir = Join-Path $TargetDir "criterion"
if ($Force -and (Test-Path $criterionDir)) {
    Remove-Item -Recurse -Force $criterionDir
}

$args = @("bench", "--bench", "indicator_benchmark")
if ($AllFeatures) {
    $args += "--all-features"
} elseif (-not [string]::IsNullOrWhiteSpace($CargoFeatures)) {
    $args += @("--features", $CargoFeatures)
}
$args += @("--", "--noplot")

Write-Host "Repo:   $RepoRoot"
Write-Host "Target: $TargetDir"
Write-Host "Out:    $Out"
if (-not [string]::IsNullOrWhiteSpace($Match)) {
    Write-Host "Match:  $Match"
}
if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
    Write-Host "Toolchain: +$Toolchain"
}
if ($AllFeatures) {
    Write-Host "Features: --all-features"
} elseif (-not [string]::IsNullOrWhiteSpace($CargoFeatures)) {
    Write-Host "Features: $CargoFeatures"
}
if (-not [string]::IsNullOrWhiteSpace($RUSTFLAGS)) {
    Write-Host "RUSTFLAGS: $RUSTFLAGS"
}
if ($OnlyBatch) {
    Write-Host "INDICATOR_BENCH_ONLY_BATCH=1"
}
if (-not [string]::IsNullOrWhiteSpace($BatchSizes)) {
    Write-Host "INDICATOR_BENCH_BATCH_SIZES=$BatchSizes"
}

$cargoArgs = @()
if (-not [string]::IsNullOrWhiteSpace($Toolchain)) {
    $cargoArgs += "+$Toolchain"
}
$cargoArgs += $args

& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) {
    throw "cargo bench failed (exit=$LASTEXITCODE)."
}

& (Join-Path $PSScriptRoot "aggregate_indicator_bench_criterion.ps1") -TargetDir $TargetDir -Match $Match -Out $Out -CargoFeatures $CargoFeatures -AllFeatures:$AllFeatures -Toolchain $Toolchain -RestrictToBenchList
