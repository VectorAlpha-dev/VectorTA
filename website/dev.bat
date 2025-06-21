@echo off
echo Running prebuild scripts...
call node scripts/run-prebuild.js

echo.
echo Starting Astro dev server...

REM Try different approaches
if exist "node_modules\.bin\astro.cmd" (
    node_modules\.bin\astro.cmd dev
) else if exist "node_modules\astro\astro.js" (
    node node_modules\astro\astro.js dev
) else (
    echo ERROR: Astro not found!
    echo Please run: npm install
)