@echo off
REM Fix Windows ESM import paths in all test files

setlocal enabledelayedexpansion

echo Fixing Windows import paths in WASM test files...

REM PowerShell script to fix all test files
powershell -Command ^
"Get-ChildItem -Path '*.js' -Filter 'test_*.js' | ForEach-Object { $content = Get-Content $_.FullName -Raw; if ($content -match 'wasm = await import\(wasmPath\);') { $newContent = $content -replace '(\s+const wasmPath = path\.join.*\r?\n)(\s+)(wasm = await import\(wasmPath\);)', '$1$2const importPath = process.platform === ''win32'' `r`n$2    ? ''file:///'' + wasmPath.replace(/\\\\/g, ''/''):`r`n$2    : wasmPath;`r`n$2wasm = await import(importPath);'; Set-Content -Path $_.FullName -Value $newContent; Write-Host \"Fixed: $($_.Name)\" } }"

echo Done!