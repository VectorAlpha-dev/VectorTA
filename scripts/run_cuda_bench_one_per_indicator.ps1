param(
    [int]$ProcessTimeoutSeconds = 120,
    [int]$VramHeadroomMB = 4096,
    [string]$TargetDir = "",
    [string]$Out = "benchmarks\\cuda_bench_one_per_indicator.json",
    [switch]$Force,
    [int]$MaxIndicators = 0
)

$ErrorActionPreference = "Stop"

function Join-PathSegments([string]$Base, [string[]]$Segments) {
    $p = $Base
    foreach ($s in $Segments) {
        if ($s -ne "") { $p = Join-Path $p $s }
    }
    return $p
}

function To-PosixPath([string]$p) {
    return ($p -replace "\\", "/")
}

function Sanitize-CriterionName([string]$s) {
    return ($s -replace "[/\\\\]", "_")
}

function Resolve-EstimatesPath([string]$CriterionRoot, [string]$Id) {
    $parts = $Id -split "/"
    if ($parts.Count -lt 3) { return $null }

    $indicator = $parts[$parts.Count - 1]
    $prefix = @($parts[0..($parts.Count - 2)])



    for ($k = 1; $k -lt $prefix.Count; $k++) {
        $groupRaw = ($prefix[0..($k - 1)] -join "/")
        $benchRaw = ($prefix[$k..($prefix.Count - 1)] -join "/")

        $groupDir = Sanitize-CriterionName $groupRaw
        $benchDir = Sanitize-CriterionName $benchRaw

        $base = Join-Path (Join-Path $CriterionRoot $groupDir) $benchDir
        if (-not (Test-Path $base)) { continue }

        $exact = Join-Path (Join-Path $base $indicator) "new\\estimates.json"
        if (Test-Path $exact) { return $exact }

        $candDirs = Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like ($indicator + "*") }
        foreach ($d in $candDirs) {
            $cand = Join-Path $d.FullName "new\\estimates.json"
            if (Test-Path $cand) { return $cand }
        }
    }

    return $null
}

function Parse-Magnitude([string]$digits, [string]$suffix) {
    $v = [int64]$digits
    if ([string]::IsNullOrEmpty($suffix)) { return $v }
    switch ($suffix.ToLowerInvariant()) {
        "k" { return $v * 1000 }
        "m" { return $v * 1000000 }
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
    if ($s -match '(?<a>\d+)e(?<exp>\d+)') {
        $a = [int64]$Matches.a
        $exp = [int]$Matches.exp
        $pow = [int64]1
        for ($i = 0; $i -lt $exp; $i++) { $pow *= 10 }
        return [int64]($a * $pow)
    }
    if ($s -match '(?<a>\d+)(?<asuf>k|m)') {
        return Parse-Magnitude $Matches.a $Matches.asuf
    }
    return [int64]100000000000
}

function Candidate-Score([string]$id) {
    $segs = $id -split "/"
    if ($segs.Length -lt 2) {
        return [int64](3 * 1000000000 + (Estimate-Elements $id))
    }
    $group = $segs[0].ToLowerInvariant()
    $indicator = $segs[$segs.Length - 1].ToLowerInvariant()
    $mid = ""
    if ($segs.Length -gt 2) {
        $mid = (($segs[1..($segs.Length - 2)]) -join "/").ToLowerInvariant()
    }

    $lc = ($group + "/" + $mid)
    $isBatch = $lc.Contains("batch")
    $isManySeries = $lc.Contains("many_series") -or $lc.Contains(($indicator + "_many/")) -or $lc.EndsWith(($indicator + "_many"))
    $isSeries = $lc.Contains("series") -and (-not $isManySeries)

    $kindWeight =
        if ($isBatch) { 0 }
        elseif ($isManySeries) { 1 }
        elseif ($isSeries) { 2 }
        else { 3 }

    $elements = Estimate-Elements $id
    return [int64]($kindWeight * 1000000000 + $elements)
}

function Read-Estimates([string]$estPath) {
    try {
        $j = Get-Content -Raw $estPath | ConvertFrom-Json
        $meanNs = [int64]$j.mean.point_estimate
        $medianNs = [int64]$j.median.point_estimate
        $stdDevNs = if ($null -ne $j.std_dev) { [int64]$j.std_dev.point_estimate } else { 0 }
        return [pscustomobject]@{
            mean_ns = $meanNs
            median_ns = $medianNs
            std_dev_ns = $stdDevNs
        }
    } catch {
        return $null
    }
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    if ($env:CARGO_TARGET_DIR) {
        $TargetDir = $env:CARGO_TARGET_DIR
    } else {
        $TargetDir = Join-Path $env:TEMP "my_project_target_cuda"
    }
}

Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:SCCACHE_DISABLE = "1"
$env:CARGO_TARGET_DIR = $TargetDir
$env:CUDA_BENCH_VRAM_HEADROOM_MB = $VramHeadroomMB.ToString()
Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_SECS -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_MS -ErrorAction SilentlyContinue

$criterionRoot = Join-Path $TargetDir "criterion"
$LogDir = Join-Path $RepoRoot "scripts\\logs\\cuda_bench_one_per_indicator"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Repo:   $RepoRoot"
Write-Host "Target: $TargetDir"
Write-Host "Timeout per bench: ${ProcessTimeoutSeconds}s"
Write-Host "VRAM headroom:     ${VramHeadroomMB}MB"
Write-Host "Criterion: $criterionRoot"
Write-Host "Logs:     $LogDir"

Write-Host "`nListing benchmarks..."
$origOutEnc = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try {
    $list = & cargo bench --bench cuda_bench --features cuda -- --list
} finally {
    [Console]::OutputEncoding = $origOutEnc
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list cuda_bench benchmarks (exit=$LASTEXITCODE)."
}

$ids = @()
foreach ($line in $list) {
    if ($line -notmatch ":\s*benchmark\s*$") { continue }
    $id = ($line -split ":", 2)[0].Trim()
    if ($id -ne "") { $ids += $id }
}

$byIndicator = @{}
foreach ($id in $ids) {
    $ind = ($id -split "/")[-1]
    if (-not $byIndicator.ContainsKey($ind)) {
        $byIndicator[$ind] = New-Object System.Collections.Generic.List[string]
    }
    $byIndicator[$ind].Add($id)
}

$indicators = $byIndicator.Keys | Sort-Object
if ($MaxIndicators -gt 0) {
    $indicators = @($indicators | Select-Object -First $MaxIndicators)
}

$outPath = Join-Path $RepoRoot $Out
$outDir = Split-Path $outPath -Parent
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$results = New-Object System.Collections.Generic.List[object]
$ok = 0
$failed = 0

foreach ($indicator in $indicators) {
    $cands = @($byIndicator[$indicator])
    $cands = $cands | Sort-Object { Candidate-Score $_ }

    $picked = $null
    $attempts = New-Object System.Collections.Generic.List[object]

    foreach ($id in $cands) {
        $est = Resolve-EstimatesPath $criterionRoot $id

        if (-not $Force -and $est -and (Test-Path $est)) {
            $e = Read-Estimates $est
            if ($null -ne $e -and $e.median_ns -gt 0) {
                $picked = [pscustomobject]@{
                    indicator = $indicator
                    id = $id
                    status = "ok"
                    mean_ns = $e.mean_ns
                    median_ns = $e.median_ns
                    std_dev_ns = $e.std_dev_ns
                    median_ms = [double]$e.median_ns / 1000000.0
                }
                break
            }
        }

        Write-Host ("run {0}: {1}" -f $indicator, $id)

        $safe = ($id -replace "[^a-zA-Z0-9_.-]", "_")
        $outLog = Join-Path $LogDir ("{0}.out.txt" -f $safe)
        $errLog = Join-Path $LogDir ("{0}.err.txt" -f $safe)



        $idArg = '"' + ($id -replace '"', '""') + '"'

        $args = @(
            "bench",
            "--bench", "cuda_bench",
            "--features", "cuda",
            "--",
            $idArg,
            "--noplot"
        )

        $p = Start-Process -FilePath "cargo" -ArgumentList $args -NoNewWindow -PassThru `
            -RedirectStandardOutput $outLog -RedirectStandardError $errLog

        $timedOut = $false
        if (-not $p.WaitForExit($ProcessTimeoutSeconds * 1000)) {
            $timedOut = $true
            & taskkill /PID $p.Id /T /F | Out-Null
        }

        $e = $null
        $est = Resolve-EstimatesPath $criterionRoot $id
        if ($est -and (Test-Path $est)) { $e = Read-Estimates $est }

        $attempts.Add([pscustomobject]@{
            id = $id
            exitcode = $p.ExitCode
            timeout = $timedOut
            has_estimates = ($est -and (Test-Path $est))
            median_ns = if ($null -ne $e) { $e.median_ns } else { 0 }
            out_log = $outLog
            err_log = $errLog
        })

        if (-not $timedOut -and $null -ne $e -and $e.median_ns -gt 0) {
            $picked = [pscustomobject]@{
                indicator = $indicator
                id = $id
                status = "ok"
                mean_ns = $e.mean_ns
                median_ns = $e.median_ns
                std_dev_ns = $e.std_dev_ns
                median_ms = [double]$e.median_ns / 1000000.0
            }
            break
        }
    }

    if ($null -eq $picked) {
        $failed += 1
        $picked = [pscustomobject]@{
            indicator = $indicator
            status = "failed"
            attempts = $attempts
        }
    } else {
        $ok += 1
    }

    $results.Add($picked)

    $outObj = [pscustomobject]@{
        generated_at = (Get-Date).ToUniversalTime().ToString("o")
        target_dir = $TargetDir
        process_timeout_seconds = $ProcessTimeoutSeconds
        vram_headroom_mb = $VramHeadroomMB
        indicator_count = $indicators.Count
        ok = $ok
        failed = $failed
        results = $results
    }
    $outObj | ConvertTo-Json -Depth 8 | Out-File -FilePath $outPath -Encoding utf8
}

Write-Host "`nDone."
Write-Host ("Wrote: {0}" -f $outPath)
Write-Host ("OK: {0}, Failed: {1}" -f $ok, $failed)
