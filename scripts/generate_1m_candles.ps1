$ErrorActionPreference = "Stop"

param(
    [string]$InputCsv = "src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv",
    [string]$OutputCsv = "src/data/1MillionCandles.csv",
    [int]$Repeats = 10
)

if ($Repeats -lt 1) {
    throw "Repeats must be >= 1"
}

if (!(Test-Path -LiteralPath $InputCsv)) {
    throw "Input CSV not found: $InputCsv"
}

$lines = New-Object System.Collections.Generic.List[string]
foreach ($line in [System.IO.File]::ReadLines($InputCsv)) {
    if (![string]::IsNullOrWhiteSpace($line)) {
        $lines.Add($line)
    }
}

if ($lines.Count -lt 3) {
    throw "Input CSV must have a header plus at least 2 data rows: $InputCsv"
}

$header = $lines[0]
$data = $lines.GetRange(1, $lines.Count - 1)

function Get-TimestampMs([string]$row) {
    $tok = $row.Split(',', 2)[0]
    return [Int64]$tok
}

$firstTs = Get-TimestampMs $data[0]
$secondTs = Get-TimestampMs $data[1]
$lastTs = Get-TimestampMs $data[$data.Count - 1]

$step = $secondTs - $firstTs
if ($step -le 0) {
    throw "Unexpected timestamp step (first=$firstTs second=$secondTs)."
}

$delta = ($lastTs - $firstTs) + $step
if ($delta -le 0) {
    throw "Unexpected timestamp delta (first=$firstTs last=$lastTs step=$step)."
}

$outDir = Split-Path -Parent $OutputCsv
if ($outDir -and !(Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

$writer = New-Object System.IO.StreamWriter($OutputCsv, $false, [System.Text.UTF8Encoding]::new($false))
try {
    $writer.WriteLine($header)

    for ($rep = 0; $rep -lt $Repeats; $rep++) {
        $offset = [Int64]$rep * $delta
        foreach ($row in $data) {
            $parts = $row.Split(',')
            $parts[0] = ([Int64]$parts[0] + $offset).ToString()
            $writer.WriteLine(($parts -join ','))
        }
    }
} finally {
    $writer.Dispose()
}

$rows = $data.Count * $Repeats
Write-Host "Wrote $OutputCsv ($rows rows) from $InputCsv (x$Repeats)."
