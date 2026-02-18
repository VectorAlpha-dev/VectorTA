param(
    [string]$Indicator = "",
    [string]$BenchId = "",
    [ValidateSet("batch", "many_series", "any")]
    [string]$Scenario = "batch",
    [string]$KernelRegex = "",
    [ValidateSet("full", "speedOfLight", "launchStats")]
    [string]$NcuSet = "full",
    [int]$LaunchCount = 1,
    [int]$LaunchSkip = 0,
    [int]$VramHeadroomMB = 4096,
    [string]$TargetDir = "",
    [string]$OutDir = "",
    [switch]$ListOnly
)

$ErrorActionPreference = "Stop"

function Parse-Magnitude([string]$digits, [string]$suffix) {
    $v = [int64]$digits
    if ([string]::IsNullOrWhiteSpace($suffix)) { return $v }
    switch ($suffix.ToLowerInvariant()) {
        "k" { return [int64]($v * 1000) }
        "m" { return [int64]($v * 1000000) }
        default { return $v }
    }
}

function Estimate-Elements([string]$id) {
    $s = $id.ToLowerInvariant().Replace("×", "x")
    $s = ($s -replace "\\s+", "")

    if ($s -match '(?<a>\d+)(?<asuf>k|m)?_x_(?<b>\d+)(?<bsuf>k|m)?') {
        $a = Parse-Magnitude $Matches.a $Matches.asuf
        $b = Parse-Magnitude $Matches.b $Matches.bsuf
        return [int64]($a * $b)
    }

    if ($s -match '(?<a>\d+)(?<asuf>k|m)?x(?<b>\d+)(?<bsuf>k|m)?') {
        $a = Parse-Magnitude $Matches.a $Matches.asuf
        $b = Parse-Magnitude $Matches.b $Matches.bsuf
        return [int64]($a * $b)
    }

    return [int64]0
}

function Scenario-Rank([string]$id, [string]$scenario) {
    $s = $id.ToLowerInvariant()
    switch ($scenario) {
        "batch" {
            if ($s.Contains("batch")) { return 0 }
            if ($s.Contains("many_series")) { return 1 }
            return 2
        }
        "many_series" {
            if ($s.Contains("many_series")) { return 0 }
            if ($s.Contains("batch")) { return 1 }
            return 2
        }
        default { return 0 }
    }
}

function To-Number([object]$v) {
    if ($null -eq $v) { return $null }
    $s = [string]$v
    if ([string]::IsNullOrWhiteSpace($s)) { return $null }
    $clean = $s.Trim().Replace(",", "")
    $out = 0.0
    if ([double]::TryParse($clean, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$out)) {
        return $out
    }
    return $null
}

function Get-MetricValue($row, [string[]]$names) {
    foreach ($n in $names) {
        if ($row.PSObject.Properties.Name -contains $n) {
            return To-Number $row.$n
        }
    }
    return $null
}

if ([string]::IsNullOrWhiteSpace($Indicator) -and [string]::IsNullOrWhiteSpace($BenchId)) {
    throw "Provide -Indicator <name> or -BenchId <criterion_id>."
}

$ncu = Get-Command ncu -ErrorAction SilentlyContinue
if ($null -eq $ncu) {
    throw "Nsight Compute CLI (ncu) not found on PATH."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $RepoRoot "target"
}
$resolvedTarget = Resolve-Path -LiteralPath $TargetDir -ErrorAction SilentlyContinue
if ($null -ne $resolvedTarget) {
    $TargetDir = $resolvedTarget.Path
}
$env:CARGO_TARGET_DIR = $TargetDir

Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:SCCACHE_DISABLE = "1"

Write-Host "Building cuda_bench binary..."
& cargo bench --bench cuda_bench --features cuda --no-run
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build cuda_bench (exit=$LASTEXITCODE)."
}

$depsDir = Join-Path $TargetDir "release\deps"
$benchExe = Get-ChildItem -LiteralPath $depsDir -Filter "cuda_bench-*.exe" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $benchExe) {
    throw "Could not locate cuda_bench executable under $depsDir."
}

$listLines = & $benchExe.FullName --list
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list cuda_bench benchmarks (exit=$LASTEXITCODE)."
}

$ids = @()
foreach ($line in $listLines) {
    if ($line -match ":\s*benchmark\s*$") {
        $ids += (($line -split ":", 2)[0].Trim())
    }
}

if ($ids.Count -eq 0) {
    throw "No benchmark IDs were discovered from cuda_bench --list."
}

if ($ListOnly) {
    if ([string]::IsNullOrWhiteSpace($Indicator)) {
        $ids | Sort-Object | ForEach-Object { Write-Host $_ }
        exit 0
    }
    $matches = $ids | Where-Object { ($_ -split "/")[-1].ToLowerInvariant() -eq $Indicator.ToLowerInvariant() } | Sort-Object
    if ($matches.Count -eq 0) {
        throw "No benchmark IDs found for indicator '$Indicator'."
    }
    $matches | ForEach-Object { Write-Host $_ }
    exit 0
}

$selectedId = $BenchId
if ([string]::IsNullOrWhiteSpace($selectedId)) {
    $cand = $ids | Where-Object { ($_ -split "/")[-1].ToLowerInvariant() -eq $Indicator.ToLowerInvariant() }
    if ($cand.Count -eq 0) {
        throw "No benchmark IDs found for indicator '$Indicator'. Use -ListOnly to inspect IDs."
    }
    $selectedId = $cand |
        Sort-Object @{ Expression = { Scenario-Rank $_ $Scenario }; Ascending = $true }, `
                    @{ Expression = { Estimate-Elements $_ }; Ascending = $false } |
        Select-Object -First 1
} else {
    if ($ids -notcontains $selectedId) {
        throw "BenchId '$selectedId' was not found in cuda_bench --list output."
    }
}

$indicatorFromId = ($selectedId -split "/")[-1]
if ([string]::IsNullOrWhiteSpace($Indicator)) {
    $Indicator = $indicatorFromId
}

if ([string]::IsNullOrWhiteSpace($KernelRegex)) {
    $KernelRegex = ".*" + [Regex]::Escape($Indicator.ToLowerInvariant()) + ".*"
}

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = Join-Path $RepoRoot ("benchmarks\\nsight\\" + $stamp + "_" + $Indicator)
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$safeId = ($selectedId -replace "[^a-zA-Z0-9_.-]", "_")
$reportBase = Join-Path $OutDir ($safeId + "_ncu")
$reportPath = $reportBase + ".ncu-rep"
$ncuLog = Join-Path $OutDir ($safeId + ".ncu.log")
$rawCsv = Join-Path $OutDir ($safeId + ".ncu.raw.csv")
$summaryJson = Join-Path $OutDir ($safeId + ".ncu.summary.json")

$env:CUDA_BENCH_WARMUP_MS = "0"
$env:CUDA_BENCH_GROUP_WARMUP_MS = "1"
$env:CUDA_BENCH_GROUP_MEASURE_MS = "1"
$env:CUDA_BENCH_MIN_SAMPLES = "10"
$env:CUDA_BENCH_SAMPLE_SIZE = "10"
$env:CUDA_BENCH_VRAM_HEADROOM_MB = $VramHeadroomMB.ToString()
Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_SECS -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_MS -ErrorAction SilentlyContinue

Write-Host "Selected benchmark: $selectedId"
Write-Host "Kernel regex:       $KernelRegex"
Write-Host "Nsight set:         $NcuSet"
Write-Host "Output dir:         $OutDir"

$ncuArgs = @(
    "--target-processes", "all",
    "--kernel-name-base", "demangled",
    "--kernel-name", "regex:$KernelRegex",
    "--launch-count", $LaunchCount.ToString(),
    "--launch-skip", $LaunchSkip.ToString(),
    "--set", $NcuSet,
    "--force-overwrite",
    "--export", $reportBase,
    $benchExe.FullName,
    $selectedId,
    "--noplot"
)

Write-Host "Running ncu..."
$ncuOutput = & $ncu.Source @ncuArgs 2>&1
$exitCode = $LASTEXITCODE
$ncuOutput | Out-File -FilePath $ncuLog -Encoding utf8
$combined = (($ncuOutput | ForEach-Object { [string]$_ }) -join "`n")

if ($combined -match "ERR_NVGPUCTRPERM") {
    throw @"
Nsight Compute could not access GPU performance counters (ERR_NVGPUCTRPERM).
Enable GPU performance counters (NVIDIA Control Panel -> Developer -> Manage GPU Performance Counters), or run the shell with sufficient privileges, then rerun.
Logs:
  $ncuLog
"@
}

if ($exitCode -ne 0 -and -not (Test-Path $reportPath)) {
    throw "ncu failed with exit code $exitCode. Check log: $ncuLog"
}

if (-not (Test-Path $reportPath)) {
    throw "ncu did not produce a report file at $reportPath"
}

& $ncu.Source --import $reportPath --csv --page raw | Out-File -FilePath $rawCsv -Encoding utf8
if ($LASTEXITCODE -ne 0) {
    throw "Failed to export ncu raw CSV from $reportPath"
}

$rows = Import-Csv -Path $rawCsv
$dataRow = $rows | Where-Object { $_.ID -match "^\d+$" } | Select-Object -Last 1

$metrics = [ordered]@{
    kernel_name = if ($null -ne $dataRow) {
        if ($dataRow.PSObject.Properties.Name -contains "Kernel Name") { $dataRow."Kernel Name" } else { $dataRow."launch__kernel_name" }
    } else {
        $null
    }
    duration_ms = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("gpu__time_duration.sum") } else { $null }
    sm_throughput_pct = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("sm__throughput.avg.pct_of_peak_sustained_elapsed") } else { $null }
    dram_throughput_pct = if ($null -ne $dataRow) {
        Get-MetricValue $dataRow @(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        )
    } else {
        $null
    }
    active_warps_pct = if ($null -ne $dataRow) {
        Get-MetricValue $dataRow @(
            "sm__warps_active.avg.pct_of_peak_sustained_active",
            "smsp__warps_active.avg.pct_of_peak_sustained_active"
        )
    } else {
        $null
    }
    block_size = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("launch__block_size") } else { $null }
    grid_size = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("launch__grid_size") } else { $null }
    registers_per_thread = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("launch__registers_per_thread") } else { $null }
    shared_mem_per_block = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("launch__shared_mem_per_block") } else { $null }
    waves_per_multiprocessor = if ($null -ne $dataRow) { Get-MetricValue $dataRow @("launch__waves_per_multiprocessor") } else { $null }
}

$heuristics = New-Object System.Collections.Generic.List[string]
$smPct = $metrics.sm_throughput_pct
$dramPct = $metrics.dram_throughput_pct
$activeWarpsPct = $metrics.active_warps_pct

if ($null -ne $smPct -and $null -ne $dramPct) {
    if ($dramPct -ge 80 -and $smPct -lt 70) {
        $heuristics.Add("Likely memory bound: DRAM throughput is high while SM throughput is lower. Check coalescing, cache reuse, and global-memory traffic.")
    } elseif ($smPct -ge 80 -and $dramPct -lt 70) {
        $heuristics.Add("Likely compute bound: SM throughput is high while DRAM throughput is lower. Check instruction mix, dependency chains, and math throughput.")
    } elseif ($smPct -lt 50 -and $dramPct -lt 50) {
        $heuristics.Add("Neither compute nor memory throughput is near peak. Check occupancy, launch configuration, and potential serialization/latency stalls.")
    }
}

if ($null -ne $activeWarpsPct -and $activeWarpsPct -lt 50) {
    $heuristics.Add("Active warps are below 50% of peak. Occupancy headroom may exist (register pressure, shared memory, or block-size tuning).")
}

if ($heuristics.Count -eq 0) {
    $heuristics.Add("No strong bottleneck signal extracted from current metrics. Open the .ncu-rep in Nsight Compute UI for section-level analysis.")
}

$summary = [ordered]@{
    generated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    indicator = $Indicator
    bench_id = $selectedId
    kernel_regex = $KernelRegex
    ncu_set = $NcuSet
    launch_count = $LaunchCount
    launch_skip = $LaunchSkip
    report_path = $reportPath
    raw_csv_path = $rawCsv
    ncu_log = $ncuLog
    metrics = $metrics
    heuristics = @($heuristics)
}

$summary | ConvertTo-Json -Depth 8 | Out-File -FilePath $summaryJson -Encoding utf8

Write-Host ""
Write-Host "Nsight profiling completed."
Write-Host "Report:   $reportPath"
Write-Host "Raw CSV:  $rawCsv"
Write-Host "Summary:  $summaryJson"
