param(
    [int]$ProcessTimeoutSeconds = 120,
    [int]$VramHeadroomMB = 4096,
    [string]$TargetDir = "",
    [string]$Match = "cuda_batch_dev/1m_x_250/",
    [switch]$Force,
    [int]$MaxBenches = 0
)

$ErrorActionPreference = "Stop"

function Join-PathSegments([string]$Base, [string[]]$Segments) {
    $p = $Base
    foreach ($s in $Segments) {
        if ($s -ne "") { $p = Join-Path $p $s }
    }
    return $p
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $env:TEMP "my_project_target_cuda"
}


Remove-Item Env:RUSTC_WRAPPER -ErrorAction SilentlyContinue
$env:SCCACHE_DISABLE = "1"

$env:CARGO_TARGET_DIR = $TargetDir
$env:CUDA_BENCH_VRAM_HEADROOM_MB = $VramHeadroomMB.ToString()



Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_SECS -ErrorAction SilentlyContinue
Remove-Item Env:CUDA_BENCH_SCENARIO_TIMEOUT_MS -ErrorAction SilentlyContinue

$LogDir = Join-Path $RepoRoot "scripts\\logs\\cuda_bench_1m_x_250"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Repo:   $RepoRoot"
Write-Host "Target: $TargetDir"
Write-Host "Match:  $Match"
Write-Host "Timeout per bench: ${ProcessTimeoutSeconds}s"
Write-Host "VRAM headroom:     ${VramHeadroomMB}MB"
Write-Host "Logs:   $LogDir"

Write-Host "`nListing benchmarks..."
$list = & cargo bench --bench cuda_bench --features cuda -- --list
if ($LASTEXITCODE -ne 0) {
    throw "Failed to list cuda_bench benchmarks (exit=$LASTEXITCODE)."
}

$ids = @()
foreach ($line in $list) {
    if ($line -notmatch ":\s*benchmark\s*$") { continue }
    $id = ($line -split ":", 2)[0].Trim()
    if ($id -like "*$Match*") {
        $ids += $id
    }
}

if ($MaxBenches -gt 0) {
    $ids = @($ids | Select-Object -First $MaxBenches)
}

if ($ids.Count -eq 0) {
    throw "No benchmarks matched '$Match'."
}

$criterionRoot = Join-Path $TargetDir "criterion"

$failures = New-Object System.Collections.Generic.List[object]

for ($i = 0; $i -lt $ids.Count; $i++) {
    $id = $ids[$i]
    $segments = $id -split "/"
    $benchDir = Join-PathSegments $criterionRoot $segments
    $estimates = Join-Path $benchDir "new\\estimates.json"

    if (-not $Force -and (Test-Path $estimates)) {
        Write-Host ("[{0}/{1}] skip (exists) {2}" -f ($i + 1), $ids.Count, $id)
        continue
    }

    Write-Host ("[{0}/{1}] run {2}" -f ($i + 1), $ids.Count, $id)

    $safe = ($id -replace "[^a-zA-Z0-9_.-]", "_")
    $outLog = Join-Path $LogDir ("{0}.out.txt" -f $safe)
    $errLog = Join-Path $LogDir ("{0}.err.txt" -f $safe)

    $args = @(
        "bench",
        "--bench", "cuda_bench",
        "--features", "cuda",
        "--",
        $id,
        "--noplot"
    )

    $p = Start-Process -FilePath "cargo" -ArgumentList $args -NoNewWindow -PassThru `
        -RedirectStandardOutput $outLog -RedirectStandardError $errLog

    if (-not $p.WaitForExit($ProcessTimeoutSeconds * 1000)) {
        Write-Warning ("timeout; killing PID={0} ({1})" -f $p.Id, $id)
        & taskkill /PID $p.Id /T /F | Out-Null
        $failures.Add([pscustomobject]@{
            id = $id
            reason = "timeout"
            out_log = $outLog
            err_log = $errLog
        })
        continue
    }



    $p.WaitForExit()
    $p.Refresh()
    $exitCode = $p.ExitCode

    if ($null -eq $exitCode -and (Test-Path $estimates)) {
        $exitCode = 0
    }

    if ($exitCode -ne 0) {
        Write-Warning ("failed exit={0} ({1})" -f $exitCode, $id)
        $failures.Add([pscustomobject]@{
            id = $id
            reason = "exitcode"
            exitcode = $exitCode
            out_log = $outLog
            err_log = $errLog
        })
        continue
    }

    if (-not (Test-Path $estimates)) {
        Write-Warning ("missing estimates.json after success ({0})" -f $id)
        $failures.Add([pscustomobject]@{
            id = $id
            reason = "missing_estimates"
            out_log = $outLog
            err_log = $errLog
        })
        continue
    }
}

$summaryPath = Join-Path $LogDir "summary.json"
$failures | ConvertTo-Json -Depth 5 | Out-File -FilePath $summaryPath -Encoding utf8

Write-Host "`nDone."
Write-Host ("Failures: {0}" -f $failures.Count)
Write-Host ("Summary:  {0}" -f $summaryPath)
