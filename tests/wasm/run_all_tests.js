#!/usr/bin/env node
/**
 * Run all WASM binding tests - equivalent to 'cargo test --features nightly-avx'
 * Usage:
 *   node run_all_tests.js              # Run all tests
 *   node run_all_tests.js alma         # Run only alma tests
 *   node run_all_tests.js --verbose    # Run with verbose output
 */
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function checkWasmBuilt() {
    const pkgDir = path.join(__dirname, '../../pkg');
    const wasmFile = path.join(pkgDir, 'my_project_bg.wasm');
    const jsFile = path.join(pkgDir, 'my_project.js');
    
    if (!fs.existsSync(wasmFile) || !fs.existsSync(jsFile)) {
        console.error('ERROR: WASM module not built!');
        console.error('Please run: wasm-pack build --features wasm --target nodejs');
        return false;
    }
    return true;
}

function installDependencies() {
    // Check if node_modules exists
    const nodeModulesPath = path.join(__dirname, 'node_modules');
    if (!fs.existsSync(nodeModulesPath)) {
        console.log('Installing test dependencies...');
        execSync('npm install', { cwd: __dirname, stdio: 'inherit' });
    }
}

async function runTests() {
    if (!checkWasmBuilt()) {
        process.exit(1);
    }
    
    installDependencies();
    
    console.log('Running WASM binding tests...');
    const startTime = Date.now();
    
    // Get all test files
    const testDir = __dirname;
    const testFiles = fs.readdirSync(testDir)
        .filter(f => f.startsWith('test_') && f.endsWith('.js'))
        .filter(f => f !== 'test_utils.js');
    
    console.log(`Found ${testFiles.length} test files`);
    
    // Build test command
    const args = process.argv.slice(2);
    let cmd = 'node --test';
    
    // Add verbose flag if requested
    if (args.includes('--verbose') || args.includes('-v')) {
        cmd += ' --test-reporter=spec';
    } else {
        cmd += ' --test-reporter=dot';
    }
    
    // Filter by pattern if provided
    const pattern = args.find(arg => !arg.startsWith('-'));
    if (pattern) {
        // Check if it's a specific test file
        const testFile = `test_${pattern}.js`;
        if (testFiles.includes(testFile)) {
            cmd += ` ${path.join(testDir, testFile)}`;
            console.log(`Running specific test file: ${testFile}`);
        } else {
            // Use as pattern
            cmd += ` --test-name-pattern="${pattern}"`;
            cmd += ' ' + testFiles.map(f => path.join(testDir, f)).join(' ');
            console.log(`Running tests matching pattern: ${pattern}`);
        }
    } else {
        // Run all test files
        cmd += ' ' + testFiles.map(f => path.join(testDir, f)).join(' ');
    }
    
    try {
        execSync(cmd, { stdio: 'inherit' });
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`\n✓ All tests passed in ${elapsed.toFixed(2)}s`);
        process.exit(0);
    } catch (error) {
        const elapsed = (Date.now() - startTime) / 1000;
        console.log(`\n✗ Tests failed after ${elapsed.toFixed(2)}s`);
        process.exit(1);
    }
}

runTests().catch(error => {
    console.error('Unexpected error:', error);
    process.exit(1);
});