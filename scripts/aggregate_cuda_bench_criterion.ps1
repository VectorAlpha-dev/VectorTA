param(
    [string]$TargetDir = "",
    [string]$Match = "cuda_batch_dev/1m_x_250/",
    [string]$Out = "benchmarks\\cuda_bench_1m_x_250_results.json",
    [switch]$RestrictToBenchList
)

$ErrorActionPreference = "Stop"

function To-PosixPath([string]$p) {
    return ($p -replace "\\", "/")
}

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    if ($env:CARGO_TARGET_DIR) {
        $TargetDir = $env:CARGO_TARGET_DIR
    } else {
        $TargetDir = Join-Path $env:TEMP "my_project_target_cuda"
    }
}

$TargetDir = [System.IO.Path]::GetFullPath($TargetDir)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$criterionDir = Join-Path $TargetDir "criterion"
if (-not (Test-Path $criterionDir)) {
    throw "criterion dir not found: $criterionDir"
}

$allowedIds = $null
if ($RestrictToBenchList) {
    Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
    $env:SCCACHE_DISABLE = "1"
    $env:CARGO_TARGET_DIR = $TargetDir

    $lines = & cargo bench --bench cuda_bench --features cuda -- --list
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list cuda_bench benchmarks (exit=$LASTEXITCODE)."
    }

    $allowedIds = @{}
    foreach ($line in $lines) {
        if ($line -notmatch ":\s*benchmark\s*$") { continue }
        $id = ($line -split ":", 2)[0].Trim()
        if ($Match -and ($id -notlike "*$Match*")) { continue }
        $allowedIds[$id] = $true
    }
}

$benchmarks = New-Object System.Collections.Generic.List[object]

$estimateFiles = Get-ChildItem -Path $criterionDir -Recurse -File -Filter "estimates.json" |
    Where-Object { $_.FullName -match '\\new\\estimates\.json$' }

foreach ($f in $estimateFiles) {
    $benchDir = Split-Path (Split-Path $f.FullName -Parent) -Parent # ...\<id>\new\estimates.json -> ...\<id>
    $idRel = $benchDir.Substring($criterionDir.Length).TrimStart("\\")
    $id = To-PosixPath $idRel
    if ($Match -and ($id -notlike "*$Match*")) {
        continue
    }
    if ($null -ne $allowedIds -and -not $allowedIds.ContainsKey($id)) {
        continue
    }

    $j = Get-Content -Raw $f.FullName | ConvertFrom-Json
    $meanNs = [int64]$j.mean.point_estimate
    $medianNs = [int64]$j.median.point_estimate
    $stdDevNs = if ($null -ne $j.std_dev) { [int64]$j.std_dev.point_estimate } else { 0 }
    $status = if ($medianNs -gt 0) { "ok" } else { "skipped" }

    $benchmarks.Add([pscustomobject]@{
        id = $id
        status = $status
        mean_ns = $meanNs
        median_ns = $medianNs
        std_dev_ns = $stdDevNs
        median_ms = [double]$medianNs / 1000000.0
    })
}

$benchmarks = $benchmarks | Sort-Object id

$outObj = [pscustomobject]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    target_dir = $TargetDir
    match = $Match
    count = $benchmarks.Count
    benchmarks = $benchmarks
}

$outPath = Join-Path $RepoRoot $Out
$outDir = Split-Path $outPath -Parent
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$outObj | ConvertTo-Json -Depth 6 | Out-File -FilePath $outPath -Encoding utf8
Write-Host ("Wrote {0} ({1} benchmarks)" -f $outPath, $benchmarks.Count)
