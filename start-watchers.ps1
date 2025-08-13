# start-watchers.ps1
$workDir = 'C:\Rust Projects\my_project'

$commands = @(
  'cargo watch -x check',
  'cargo watch --features nightly-avx -x check',
  'cargo watch --features python -x check',
  'cargo watch --features wasm -x check'
)

foreach ($cmd in $commands) {
  Start-Process -FilePath 'powershell.exe' `
    -WorkingDirectory $workDir `
    -ArgumentList '-NoExit','-NoLogo','-Command', $cmd
}