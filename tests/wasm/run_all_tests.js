#!/usr/bin/env node
/**
 * Run all WASM binding tests - equivalent to 'cargo test --features nightly-avx'
 * Usage:
 *   node run_all_tests.js              # Run all tests
 *   node run_all_tests.js alma         # Run only alma tests
 *   node run_all_tests.js --verbose    # Run with verbose output
 */
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function checkWasmBuilt() {
    const pkgDir = path.join(__dirname, '../../pkg');
    const wasmFile = path.join(pkgDir, 'vector_ta_bg.wasm');
    const jsFile = path.join(pkgDir, 'vector_ta.js');
    
    if (!fs.existsSync(wasmFile) || !fs.existsSync(jsFile)) {
        console.error('ERROR: WASM module not built!');
        console.error('Please run: wasm-pack build --features wasm --target nodejs');
        return false;
    }
    return true;
}

function installDependencies() {
    
    
    const nodeModulesPath = path.join(__dirname, 'node_modules');
    if (!fs.existsSync(nodeModulesPath)) {
        console.log('[wasm-tests] Skipping npm install (no external deps required)');
    }
}

async function runTests() {
    if (!checkWasmBuilt()) {
        process.exit(1);
    }
    
    installDependencies();
    
    console.log('Running WASM binding tests...');
    const startTime = Date.now();
    
    
    const testDir = __dirname;
    const testFiles = fs.readdirSync(testDir)
        .filter(f => f.startsWith('test_') && f.endsWith('.js'))
        .filter(f => f !== 'test_utils.js');
    
    console.log(`Found ${testFiles.length} test files`);
    
    
    const args = process.argv.slice(2);
    let cmd = 'node --experimental-wasm-modules --test';
    
    
    if (args.includes('--verbose') || args.includes('-v')) {
        cmd += ' --test-reporter=spec';
    } else {
        cmd += ' --test-reporter=dot';
    }
    
    
    const pattern = args.find(arg => !arg.startsWith('-'));
    if (pattern) {
        
        const testFile = `test_${pattern}.js`;
        if (testFiles.includes(testFile)) {
            cmd += ` "${path.join(testDir, testFile)}"`;
            console.log(`Running specific test file: ${testFile}`);
        } else {
            
            cmd += ` --test-name-pattern="${pattern}"`;
            cmd += ' ' + testFiles.map(f => `"${path.join(testDir, f)}"`).join(' ');
            console.log(`Running tests matching pattern: ${pattern}`);
        }
    } else {
        
        cmd += ' ' + testFiles.map(f => `"${path.join(testDir, f)}"`).join(' ');
    }
    
    try {
        
        const batchSize = 5; 
        const testFilePaths = testFiles.map(f => path.join(testDir, f));
        
        if (pattern && testFiles.includes(`test_${pattern}.js`)) {
            
            execSync(cmd, { stdio: 'inherit' });
        } else if (testFilePaths.length <= batchSize) {
            
            execSync(cmd, { stdio: 'inherit' });
        } else {
            
            for (let i = 0; i < testFilePaths.length; i += batchSize) {
                const batch = testFilePaths.slice(i, i + batchSize);
                const batchCmd = `node --experimental-wasm-modules --test ${args.includes('--verbose') || args.includes('-v') ? '--test-reporter=spec' : '--test-reporter=dot'} ${batch.map(f => `"${f}"`).join(' ')}`;
                
                if (i > 0) {
                    process.stdout.write('\n'); 
                }
                
                execSync(batchCmd, { stdio: 'inherit' });
            }
        }
        
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
