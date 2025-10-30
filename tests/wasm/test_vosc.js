/**
 * WASM binding tests for VOSC indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { 
    loadTestData, 
    assertArrayClose, 
    assertClose,
    isNaN,
    assertAllNaN,
    assertNoNaN,
    EXPECTED_OUTPUTS 
} from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let wasm;
let testData;

test.before(async () => {
    // Load a known-good NodeJS wrapper to avoid ESM/"env" import issues
    try {
        const req = createRequire(import.meta.url);
        // Use the full CommonJS wrapper copy to avoid ESM 'module' issues
        const mod = req(path.join(__dirname, 'my_project_full.cjs'));
        wasm = mod; // CommonJS exports
    } catch (error) {
        console.error('Failed to load local WASM wrapper (tests/wasm/my_project.js).');
        console.error('If missing, run: wasm-pack build -- --features wasm --no-default-features');
        throw error;
    }

    testData = loadTestData();
});

test('VOSC accuracy', () => {
    // Test with default parameters from Python test
    const volume = testData.volume;
    const shortPeriod = 2;
    const longPeriod = 5;
    
    // Run VOSC calculation
    const result = wasm.vosc_js(volume, shortPeriod, longPeriod);
    
    // Expected values from Python test
    const expectedLastFive = [
        -39.478510754298895,
        -25.886077312645188,
        -21.155087549723756,
        -12.36093768813373,
        48.70809369473075,
    ];
    
    // Verify output length
    assert.strictEqual(result.length, volume.length, 'VOSC length mismatch');
    
    // Check last 5 values
    const startIdx = result.length - 5;
    for (let i = 0; i < expectedLastFive.length; i++) {
        assertClose(result[startIdx + i], expectedLastFive[i], 0.1, 
            `VOSC mismatch at index ${startIdx + i}`);
    }
    
    // Check warmup period (should be NaN)
    const warmupPeriod = longPeriod - 1;
    for (let i = 0; i < warmupPeriod; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i}`);
    }
});

test('VOSC error handling', () => {
    // Test empty data
    assert.throws(() => {
        wasm.vosc_js([], 2, 5);
    }, 'Should throw on empty data');
    
    // Test zero short period
    assert.throws(() => {
        wasm.vosc_js([1.0, 2.0, 3.0], 0, 5);
    }, 'Should throw on zero short period');
    
    // Test zero long period
    assert.throws(() => {
        wasm.vosc_js([1.0, 2.0, 3.0], 2, 0);
    }, 'Should throw on zero long period');
    
    // Test short period > long period
    assert.throws(() => {
        wasm.vosc_js([1.0, 2.0, 3.0, 4.0, 5.0], 5, 2);
    }, 'Should throw when short period > long period');
    
    // Test all NaN values
    assert.throws(() => {
        wasm.vosc_js([NaN, NaN, NaN], 2, 3);
    }, 'Should throw on all NaN values');
});

test('VOSC fast API', () => {
    const volume = testData.volume;
    const shortPeriod = 2;
    const longPeriod = 5;
    
    // Allocate memory
    const len = volume.length;
    const inPtr = wasm.vosc_alloc(len);
    const outPtr = wasm.vosc_alloc(len);
    
    try {
        // Copy data to WASM memory
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        wasmMemory.set(volume, inPtr / 8);
        
        // Run calculation
        wasm.vosc_into(inPtr, outPtr, len, shortPeriod, longPeriod);
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = Array.from(result);
        
        // Compare with safe API
        const safeResult = wasm.vosc_js(volume, shortPeriod, longPeriod);
        assertArrayClose(resultCopy, safeResult, 1e-10, 
            'Fast API should match safe API');
    } finally {
        // Clean up
        wasm.vosc_free(inPtr, len);
        wasm.vosc_free(outPtr, len);
    }
});

test('VOSC in-place operation', () => {
    const volume = testData.volume;
    const shortPeriod = 2;
    const longPeriod = 5;
    
    // Allocate memory and copy data
    const len = volume.length;
    const ptr = wasm.vosc_alloc(len);
    
    try {
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        wasmMemory.set(volume, ptr / 8);
        
        // Run in-place calculation
        wasm.vosc_into(ptr, ptr, len, shortPeriod, longPeriod);
        
        // Read results
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const resultCopy = Array.from(result);
        
        // Compare with safe API
        const safeResult = wasm.vosc_js(volume, shortPeriod, longPeriod);
        assertArrayClose(resultCopy, safeResult, 1e-10, 
            'In-place operation should match safe API');
    } finally {
        wasm.vosc_free(ptr, len);
    }
});

test('VOSC batch API', () => {
    const volume = testData.volume;
    
    // Test batch calculation with small ranges
    const config = {
        short_period_range: [2, 4, 1],
        long_period_range: [5, 7, 1]
    };
    
    const result = wasm.vosc_batch(volume, config);
    
    // Verify structure
    assert.ok(result.values, 'Batch result should have values');
    assert.ok(result.combos, 'Batch result should have combos');
    assert.ok(result.rows > 0, 'Batch result should have rows');
    assert.strictEqual(result.cols, volume.length, 'Batch cols should match data length');
    
    // Verify dimensions
    const expectedCombos = 3 * 3; // 3 short periods * 3 long periods (filtered for valid combos)
    assert.ok(result.combos.length > 0, 'Should have valid combinations');
    assert.strictEqual(result.values.length, result.rows * result.cols, 
        'Values array size should match dimensions');
});

test.after(() => {
    console.log('VOSC WASM tests completed');
});
