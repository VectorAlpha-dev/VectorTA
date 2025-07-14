/**
 * WASM binding tests for VPWMA indicator.
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
import { compareWithRust } from './rust-comparison.js';

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

test('VPWMA partial params', () => {
    // Test with default parameters - mirrors check_vpwma_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
});

test('VPWMA accuracy', async () => {
    // Test VPWMA matches expected values from Rust tests - mirrors check_vpwma_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.vpwma;
    
    const result = wasm.vpwma_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.power
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-4,  // Using 1e-4 as per Rust test which uses 1e-2
        "VPWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('vpwma', result, 'close', expected.defaultParams);
});

test('VPWMA zero period', () => {
    // Test VPWMA fails with zero period - mirrors check_vpwma_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(inputData, 0, 0.382);
    }, /Invalid period/);
});

test('VPWMA period exceeds length', () => {
    // Test VPWMA fails when period exceeds data length - mirrors check_vpwma_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(dataSmall, 10, 0.382);
    }, /Invalid period/);
});

test('VPWMA very small dataset', () => {
    // Test VPWMA fails with insufficient data - mirrors check_vpwma_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.vpwma_js(singlePoint, 2, 0.382);
    }, /Invalid period|Not enough valid data/);
});

test('VPWMA empty input', () => {
    // Test VPWMA fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.vpwma_js(empty, 14, 0.382);
    }, /Input data slice is empty/);
});

test('VPWMA invalid power', () => {
    // Test VPWMA fails with invalid power - mirrors vpwma power validation
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // NaN power
    assert.throws(() => {
        wasm.vpwma_js(data, 2, NaN);
    }, /Invalid power/);
    
    // Infinite power
    assert.throws(() => {
        wasm.vpwma_js(data, 2, Infinity);
    }, /Invalid power/);
});

test('VPWMA reinput', () => {
    // Test VPWMA applied twice (re-input) - mirrors check_vpwma_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply VPWMA to VPWMA output
    const secondResult = wasm.vpwma_js(firstResult, 5, 0.5);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // Check that values after warmup are not NaN
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
        }
    }
});

test('VPWMA NaN handling', () => {
    // Test VPWMA handles NaN values correctly - mirrors check_vpwma_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.vpwma_js(close, 14, 0.382);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (50), no NaN values should exist
    if (result.length > 50) {
        for (let i = 50; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('VPWMA all NaN input', () => {
    // Test VPWMA with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.vpwma_js(allNaN, 14, 0.382);
    }, /All values are NaN/);
});

test('VPWMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14, power=0.382
    const batchResult = wasm.vpwma_batch_js(
        close,
        14, 14, 0,      // period range
        0.382, 0.382, 0  // power range
    );
    
    // Should match single calculation
    const singleResult = wasm.vpwma_js(close, 14, 0.382);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VPWMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 14, 16, 18
    const batchResult = wasm.vpwma_batch_js(
        close,
        14, 18, 2,      // period range
        0.382, 0.382, 0 // power range  
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [14, 16, 18];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vpwma_js(close, periods[i], 0.382);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VPWMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.vpwma_batch_metadata_js(
        14, 18, 2,      // period: 14, 16, 18
        0.3, 0.5, 0.1   // power: 0.3, 0.4, 0.5
    );
    
    // Should have 3 * 3 = 9 combinations
    // Each combo has 2 values: [period, power]
    assert.strictEqual(metadata.length, 9 * 2);
    
    // Check first combination
    assert.strictEqual(metadata[0], 14);   // period
    assert.strictEqual(metadata[1], 0.3);  // power
    
    // Check last combination
    assert.strictEqual(metadata[16], 18);  // period
    assert.strictEqual(metadata[17], 0.5); // power
});

test('VPWMA batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.vpwma_batch_js(
        close,
        12, 14, 2,      // 2 periods
        0.3, 0.4, 0.1   // 2 powers
    );
    
    const metadata = wasm.vpwma_batch_metadata_js(
        12, 14, 2,
        0.3, 0.4, 0.1
    );
    
    // Should have 2 * 2 = 4 combinations
    const numCombos = metadata.length / 2;
    assert.strictEqual(numCombos, 4);
    assert.strictEqual(batchResult.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo * 2];
        const power = metadata[combo * 2 + 1];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('VPWMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.vpwma_batch_js(
        close,
        5, 5, 1,
        0.382, 0.382, 0.1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.vpwma_batch_js(
        close,
        5, 7, 10, // Step larger than range
        0.382, 0.382, 0
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.vpwma_batch_js(
            new Float64Array([]),
            14, 14, 0,
            0.382, 0.382, 0
        );
    }, /Input data slice is empty/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('VPWMA WASM tests completed');
});