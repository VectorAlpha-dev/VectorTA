/**
 * WASM binding tests for BOP indicator.
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

test('BOP partial params', () => {
    // Test with standard parameters - mirrors check_bop_partial_params
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // BOP has no parameters, just OHLC inputs
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
});

test('BOP accuracy', async () => {
    // Test BOP matches expected values from Rust tests - mirrors check_bop_accuracy
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.bop;
    
    const result = wasm.bop_js(open, high, low, close);
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-10,
        "BOP last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('bop', result, 'ohlc', expected.defaultParams);
});

test('BOP default candles', () => {
    // Test BOP with default parameters - mirrors check_bop_default_candles
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
});

test('BOP with empty data', () => {
    // Test BOP fails with empty data - mirrors check_bop_with_empty_data
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.bop_js(empty, empty, empty, empty);
    }, /Data is empty/);
});

test('BOP with inconsistent lengths', () => {
    // Test BOP fails with inconsistent input lengths - mirrors check_bop_with_inconsistent_lengths
    const open = new Float64Array([1.0, 2.0, 3.0]);
    const high = new Float64Array([1.5, 2.5]);  // Wrong length
    const low = new Float64Array([0.8, 1.8, 2.8]);
    const close = new Float64Array([1.2, 2.2, 3.2]);
    
    assert.throws(() => {
        wasm.bop_js(open, high, low, close);
    }, /Inconsistent lengths/);
});

test('BOP very small dataset', () => {
    // Test BOP with single data point - mirrors check_bop_very_small_dataset
    const open = new Float64Array([10.0]);
    const high = new Float64Array([12.0]);
    const low = new Float64Array([9.5]);
    const close = new Float64Array([11.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, 1);
    // (11.0 - 10.0) / (12.0 - 9.5) = 1.0 / 2.5 = 0.4
    assertClose(result[0], 0.4, 1e-10, "BOP single value calculation");
});

test('BOP with slice data reinput', () => {
    // Test BOP with slice data re-input - mirrors check_bop_with_slice_data_reinput
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.bop_js(open, high, low, close);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - use first result as close, zeros for others
    const dummy = new Float64Array(firstResult.length);
    const secondResult = wasm.bop_js(dummy, dummy, dummy, firstResult);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // All values should be 0.0 since (first_result - 0) / (0 - 0) = 0.0
    for (let i = 0; i < secondResult.length; i++) {
        assertClose(secondResult[i], 0.0, 1e-15, 
                   `Expected BOP=0.0 for dummy data at idx ${i}`);
    }
});

test('BOP NaN handling', () => {
    // Test BOP handles values correctly without NaN - mirrors check_bop_nan_handling
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.bop_js(open, high, low, close);
    assert.strictEqual(result.length, close.length);
    
    // BOP should not produce NaN values after any warmup period
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Actually, BOP has no warmup period - it calculates from the first value
    assertNoNaN(result, "BOP should not produce any NaN values");
});

test('BOP zero range handling', () => {
    // Test BOP when high equals low (zero range)
    // When high == low, BOP should return 0.0
    const open = new Float64Array([10.0, 20.0, 30.0]);
    const high = new Float64Array([15.0, 25.0, 35.0]);
    const low = new Float64Array([15.0, 25.0, 35.0]);  // Same as high
    const close = new Float64Array([15.0, 25.0, 35.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // All values should be 0.0 since denominator is 0
    for (let i = 0; i < result.length; i++) {
        assertClose(result[i], 0.0, 1e-15, 
                   `Expected BOP=0.0 when high=low at idx ${i}`);
    }
});

test('BOP batch single', () => {
    // Test batch processing with BOP (no parameters)
    const open = new Float64Array(testData.open);
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // BOP has no parameters, so batch is just regular calculation
    const batchResult = wasm.bop_batch_js(open, high, low, close);
    
    // Should match single calculation
    const singleResult = wasm.bop_js(open, high, low, close);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('BOP batch metadata', () => {
    // Test metadata function returns empty array (no parameters)
    const metadata = wasm.bop_batch_metadata_js();
    
    // BOP has no parameters, so metadata should be empty
    assert.strictEqual(metadata.length, 0);
});

test('BOP batch unified API', () => {
    // Test unified batch API with config object
    const open = new Float64Array(testData.open.slice(0, 100));
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // BOP has no parameters, but we pass empty config for API consistency
    const config = {};  // Empty config for BOP
    
    const result = wasm.bop_batch(open, high, low, close, config);
    
    // Should have structure with values, rows, cols
    assert(result.values);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    
    // Values should match regular calculation
    const regularResult = wasm.bop_js(open, high, low, close);
    assertArrayClose(result.values, regularResult, 1e-10, "Unified API mismatch");
});

test('BOP extreme values', () => {
    // Test BOP with extreme price movements
    const open = new Float64Array([100.0, 1000.0, 10.0]);
    const high = new Float64Array([200.0, 2000.0, 20.0]);
    const low = new Float64Array([50.0, 500.0, 5.0]);
    const close = new Float64Array([150.0, 1500.0, 15.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // Calculate expected values manually
    // BOP = (close - open) / (high - low)
    const expected = [
        (150.0 - 100.0) / (200.0 - 50.0),   // 50/150 = 0.333...
        (1500.0 - 1000.0) / (2000.0 - 500.0), // 500/1500 = 0.333...
        (15.0 - 10.0) / (20.0 - 5.0)         // 5/15 = 0.333...
    ];
    
    assertArrayClose(result, expected, 1e-10, "Extreme values mismatch");
});

test('BOP negative values', () => {
    // Test BOP when close < open (negative BOP)
    const open = new Float64Array([100.0, 200.0, 300.0]);
    const high = new Float64Array([110.0, 210.0, 310.0]);
    const low = new Float64Array([80.0, 180.0, 280.0]);
    const close = new Float64Array([90.0, 190.0, 290.0]);
    
    const result = wasm.bop_js(open, high, low, close);
    
    // All should be negative: (90-100)/(110-80) = -10/30 = -0.333...
    for (let i = 0; i < result.length; i++) {
        assert(result[i] < 0, `Expected negative BOP at index ${i}`);
        assertClose(result[i], -1/3, 1e-10, `BOP value at index ${i}`);
    }
});

if (process.argv.includes('--run')) {
    // This allows running the file directly with node
    test.run();
}