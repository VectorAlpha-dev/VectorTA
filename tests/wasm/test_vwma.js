/**
 * WASM binding tests for VWMA indicator.
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

test('VWMA partial params', () => {
    // Test with default parameters - mirrors check_vwma_partial_params
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    // Test with custom period
    const result_custom = wasm.vwma_js(close, volume, 10);
    assert.strictEqual(result_custom.length, close.length);
});

test('VWMA accuracy', async () => {
    // Test VWMA matches expected values from Rust tests - mirrors check_vwma_accuracy
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    const expected = EXPECTED_OUTPUTS.vwma;
    
    const result = wasm.vwma_js(
        close,
        volume,
        expected.defaultParams.period
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-3,  // VWMA uses 1e-3 tolerance in Rust tests
        "VWMA last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('vwma', result, 'close', expected.defaultParams);
});

test('VWMA price volume mismatch', () => {
    // Test VWMA fails when price and volume lengths don't match
    const prices = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);  // Shorter array
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 3);
    }, /Price and volume mismatch/);
});

test('VWMA invalid period', () => {
    // Test VWMA fails with invalid period
    const prices = new Float64Array([10.0, 20.0, 30.0]);
    const volumes = new Float64Array([100.0, 200.0, 300.0]);
    
    // Period = 0
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 0);
    }, /Invalid period/);
    
    // Period exceeds length
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 10);
    }, /Invalid period/);
});

test('VWMA all NaN values', () => {
    // Test VWMA fails when all values are NaN
    const prices = new Float64Array(10);
    const volumes = new Float64Array(10);
    prices.fill(NaN);
    volumes.fill(NaN);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /All/);
});

test('VWMA not enough valid data', () => {
    // Test VWMA fails with insufficient valid data
    // First 8 values are NaN, only 2 valid values for period=5
    const prices = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 10.0, 20.0]);
    const volumes = new Float64Array([NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 100.0, 200.0]);
    
    assert.throws(() => {
        wasm.vwma_js(prices, volumes, 5);
    }, /Not enough valid/);
});

test('VWMA with default candles', () => {
    // Test VWMA with default parameters - mirrors check_vwma_input_with_default_candles
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Default period is 20
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
});

test('VWMA candles plus prices', () => {
    // Test VWMA with custom prices - mirrors check_vwma_candles_plus_prices
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Use slightly modified prices
    const custom_prices = close.map(v => v * 1.001);
    
    const result = wasm.vwma_js(custom_prices, volume, 20);
    assert.strictEqual(result.length, custom_prices.length);
});

test('VWMA slice reinput', () => {
    // Test VWMA applied twice (re-input) - mirrors check_vwma_slice_data_reinput
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // First pass
    const firstResult = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - use VWMA output as prices, keep same volumes
    const secondResult = wasm.vwma_js(firstResult, volume, 10);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup, should have valid values
    const start = 20 + 10 - 2;  // first period + second period - 2
    for (let i = start; i < secondResult.length; i++) {
        assert(!isNaN(secondResult[i]), `Unexpected NaN at index ${i}`);
    }
});

test('VWMA NaN handling', () => {
    // Test VWMA handles NaN values correctly
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    const result = wasm.vwma_js(close, volume, 20);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN (warmup)
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // After warmup period, no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('VWMA batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    const volume = new Float64Array(testData.volume);
    
    // Single parameter set: period=20
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        20, 20, 0  // period range
    );
    
    // Should match single calculation
    const singleResult = wasm.vwma_js(close, volume, 20);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('VWMA batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    const volume = new Float64Array(testData.volume.slice(0, 100));
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.vwma_batch_js(
        close,
        volume,
        10, 20, 5  // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.vwma_js(close, volume, periods[i]);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('VWMA batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.vwma_batch_metadata_js(
        10, 30, 5  // period: 10, 15, 20, 25, 30
    );
    
    // Should have 5 combinations
    assert.strictEqual(metadata.length, 5);
    
    // Check values
    assert.strictEqual(metadata[0], 10);
    assert.strictEqual(metadata[1], 15);
    assert.strictEqual(metadata[2], 20);
    assert.strictEqual(metadata[3], 25);
    assert.strictEqual(metadata[4], 30);
});

test('VWMA batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const volume = new Float64Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    
    // Single value sweep
    const singleBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 5, 1
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.vwma_batch_js(
        close,
        volume,
        5, 7, 10  // Step larger than range
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.vwma_batch_js(
            new Float64Array([]),
            new Float64Array([]),
            10, 10, 0
        );
    }, /All/);
});

// Note: Streaming tests would require streaming functions to be exposed in WASM bindings

test.after(() => {
    console.log('VWMA WASM tests completed');
});
