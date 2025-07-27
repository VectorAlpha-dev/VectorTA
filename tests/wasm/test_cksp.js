/**
 * WASM binding tests for CKSP indicator.
 * These tests mirror the Rust unit tests to ensure WASM bindings work correctly.
 */
import test from 'node:test';
import assert from 'node:assert';
import path from 'path';
import { fileURLToPath } from 'url';
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
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('CKSP accuracy', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    // Run CKSP with default parameters
    const result = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
    
    // Result should have 2x the length (long_values, short_values)
    assert.strictEqual(result.length, close.length * 2);
    
    // Split the result into long and short arrays
    const len = close.length;
    const longResult = result.slice(0, len);
    const shortResult = result.slice(len);
    
    // Check last 5 values match expected for long
    const longLast5 = longResult.slice(-5);
    expected['long_last_5_values'].forEach((expectedVal, i) => {
        assertClose(longLast5[i], expectedVal, 1e-5, `CKSP long value mismatch at index ${i}`);
    });
    
    // Check last 5 values match expected for short
    const shortLast5 = shortResult.slice(-5);
    expected['short_last_5_values'].forEach((expectedVal, i) => {
        assertClose(shortLast5[i], expectedVal, 1e-5, `CKSP short value mismatch at index ${i}`);
    });
});

test('CKSP error handling', () => {
    // Test with empty data
    assert.throws(() => {
        wasm.cksp_js(new Float64Array(), new Float64Array(), new Float64Array(), 10, 1.0, 9);
    }, /Data is empty/);
    
    // Test with inconsistent lengths
    assert.throws(() => {
        wasm.cksp_js(
            new Float64Array([1, 2, 3]),
            new Float64Array([1, 2]),  // Wrong length
            new Float64Array([1, 2, 3]),
            10, 1.0, 9
        );
    }, /Inconsistent/);
    
    // Test with zero period
    const high = new Float64Array([10, 11, 12]);
    const low = new Float64Array([9, 10, 10.5]);
    const close = new Float64Array([9.5, 10.5, 11]);
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 0, 1.0, 9);
    }, /Invalid param/);
    
    // Test with period exceeding data length
    assert.throws(() => {
        wasm.cksp_js(high, low, close, 10, 1.0, 9);
    }, /Not enough data/);
});

test('CKSP nan handling', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    const result = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
    
    // Split the result
    const len = close.length;
    const longResult = result.slice(0, len);
    const shortResult = result.slice(len);
    
    // Check warmup period has NaN values
    const warmupPeriod = params.p + params.q - 1;
    for (let i = 0; i < warmupPeriod && i < longResult.length; i++) {
        assert(isNaN(longResult[i]), `Expected NaN in long warmup at index ${i}`);
        assert(isNaN(shortResult[i]), `Expected NaN in short warmup at index ${i}`);
    }
    
    // After warmup, no NaN values should exist (if we have enough data)
    if (longResult.length > 240) {
        for (let i = 240; i < longResult.length; i++) {
            assert(!isNaN(longResult[i]), `Found unexpected NaN in long at index ${i}`);
            assert(!isNaN(shortResult[i]), `Found unexpected NaN in short at index ${i}`);
        }
    }
});

test('CKSP fast API (cksp_into)', () => {
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const len = close.length;
    const expected = EXPECTED_OUTPUTS['cksp'];
    const params = expected['default_params'];
    
    // Allocate output buffers
    const longPtr = wasm.cksp_alloc(len);
    const shortPtr = wasm.cksp_alloc(len);
    
    try {
        // Call the fast API
        wasm.cksp_into(
            high.buffer, 
            low.buffer, 
            close.buffer,
            longPtr,
            shortPtr,
            len,
            params.p,
            params.x,
            params.q
        );
        
        // Read results from memory
        const longResult = new Float64Array(wasm.memory.buffer, longPtr, len);
        const shortResult = new Float64Array(wasm.memory.buffer, shortPtr, len);
        
        // Verify results match the safe API
        const safeResult = wasm.cksp_js(high, low, close, params.p, params.x, params.q);
        const safeLong = safeResult.slice(0, len);
        const safeShort = safeResult.slice(len);
        
        assertArrayClose(
            Array.from(longResult),
            Array.from(safeLong),
            1e-10,
            'Fast API long values should match safe API'
        );
        
        assertArrayClose(
            Array.from(shortResult),
            Array.from(safeShort),
            1e-10,
            'Fast API short values should match safe API'
        );
        
    } finally {
        // Clean up allocated memory
        wasm.cksp_free(longPtr, len);
        wasm.cksp_free(shortPtr, len);
    }
});

test('CKSP batch processing', async () => {
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch with single parameter set
    const config = {
        p_range: [10, 10, 0],
        x_range: [1.0, 1.0, 0.0],
        q_range: [9, 9, 0]
    };
    
    const result = wasm.cksp_batch(high, low, close, config);
    
    // Check result structure
    assert(result.long_values, 'Missing long_values in batch result');
    assert(result.short_values, 'Missing short_values in batch result');
    assert(result.combos, 'Missing combos in batch result');
    assert.strictEqual(result.rows, 1, 'Expected 1 row for single parameter set');
    assert.strictEqual(result.cols, 100, 'Expected 100 columns');
    
    // Test with multiple parameter sets
    const multiConfig = {
        p_range: [5, 15, 5],  // 5, 10, 15
        x_range: [0.5, 1.5, 0.5],  // 0.5, 1.0, 1.5
        q_range: [5, 10, 5]  // 5, 10
    };
    
    const multiResult = wasm.cksp_batch(high, low, close, multiConfig);
    
    // Should have 3 * 3 * 2 = 18 combinations
    assert.strictEqual(multiResult.rows, 18, 'Expected 18 parameter combinations');
    assert.strictEqual(multiResult.combos.length, 18, 'Expected 18 combo entries');
    assert.strictEqual(multiResult.long_values.length, 18 * 100, 'Expected correct long_values length');
    assert.strictEqual(multiResult.short_values.length, 18 * 100, 'Expected correct short_values length');
});

test.after(() => {
    console.log('CKSP WASM tests completed');
});
