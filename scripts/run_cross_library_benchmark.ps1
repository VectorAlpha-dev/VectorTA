param(
    [string]$TargetDir = "",
    [string]$Out = "benchmarks\\cross_library\\benchmark_results.json",
    [string]$CargoFeatures = "",
    [switch]$AllFeatures,
    [string]$RUSTFLAGS = "",
    [string]$IndicatorFilter = "",
    [switch]$NoSortAlpha,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    if ($env:CARGO_TARGET_DIR) {
        $TargetDir = $env:CARGO_TARGET_DIR
    } else {
        $TargetDir = Join-Path $RepoRoot "target-bench-cross-library"
    }
} elseif (-not [System.IO.Path]::IsPathRooted($TargetDir)) {
    $TargetDir = Join-Path $RepoRoot $TargetDir
}

$outPath =
    if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $RepoRoot $Out }

Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:SCCACHE_DISABLE = "1"
$env:CARGO_TARGET_DIR = $TargetDir

if ([string]::IsNullOrWhiteSpace($RUSTFLAGS)) {
    Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
} else {
    $env:RUSTFLAGS = $RUSTFLAGS
}

if ([string]::IsNullOrWhiteSpace($IndicatorFilter)) {
    Remove-Item Env:IND_FILTER -ErrorAction SilentlyContinue
} else {
    $env:IND_FILTER = $IndicatorFilter
}

if ($NoSortAlpha) {
    $env:SORT_ALPHA = "0"
} else {
    $env:SORT_ALPHA = "1"
}

$crossDir = Join-Path $RepoRoot "benchmarks\\cross_library"
if (-not (Test-Path $crossDir)) {
    throw "cross_library dir not found: $crossDir"
}

$criterionDir = Join-Path $TargetDir "criterion"
if ($Force -and (Test-Path $criterionDir)) {
    Remove-Item -Recurse -Force $criterionDir
}

$generatedJsonPath = Join-Path $crossDir "benchmark_results.json"
if ($Force -and (Test-Path $generatedJsonPath)) {
    Remove-Item -Force $generatedJsonPath
}

Write-Host "Repo:   $RepoRoot"
Write-Host "CWD:    $crossDir"
Write-Host "Target: $TargetDir"
Write-Host "Out:    $outPath"
if ($AllFeatures) {
    Write-Host "Features: --all-features"
} elseif (-not [string]::IsNullOrWhiteSpace($CargoFeatures)) {
    Write-Host "Features: $CargoFeatures"
}
if (-not [string]::IsNullOrWhiteSpace($IndicatorFilter)) {
    Write-Host "IND_FILTER: $IndicatorFilter"
}
if ($NoSortAlpha) {
    Write-Host "SORT_ALPHA: 0"
}
if (-not [string]::IsNullOrWhiteSpace($RUSTFLAGS)) {
    Write-Host "RUSTFLAGS: $RUSTFLAGS"
}

Push-Location $crossDir
try {
    $args = @("bench", "--bench", "cross_library_comparison")
    if ($AllFeatures) {
        $args += "--all-features"
    } elseif (-not [string]::IsNullOrWhiteSpace($CargoFeatures)) {
        $args += @("--features", $CargoFeatures)
    }
    $args += @("--", "--noplot")

    & cargo @args
    if ($LASTEXITCODE -ne 0) {
        throw "cargo bench failed (exit=$LASTEXITCODE)."
    }
} finally {
    Pop-Location
}

if (-not (Test-Path $generatedJsonPath)) {
    throw "Expected JSON output not found: $generatedJsonPath"
}

$outDir = Split-Path $outPath -Parent
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$src = (Resolve-Path -LiteralPath $generatedJsonPath).Path
$dst =
    if (Test-Path -LiteralPath $outPath) { (Resolve-Path -LiteralPath $outPath).Path }
    else { [System.IO.Path]::GetFullPath($outPath) }
if ($src -ne $dst) { Copy-Item -Force $generatedJsonPath $outPath }

Write-Host "Wrote $outPath"
