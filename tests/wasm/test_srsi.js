/**
 * WASM binding tests for SRSI indicator.
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

test('SRSI partial params', () => {
    // Test with default parameters - mirrors check_srsi_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 14, 14, 3, 3);
    assert.strictEqual(result.length, close.length * 2); // k and d values flattened
});

test('SRSI accuracy', () => {
    // Test SRSI matches expected values from Rust tests - mirrors check_srsi_accuracy
    const close = new Float64Array(testData.close);
    
    // Expected values from Rust tests
    const expected_k = [
        65.52066633236464,
        61.22507053191985,
        57.220471530042644,
        64.61344854988147,
        60.66534359318523,
    ];
    const expected_d = [
        64.33503158970049,
        64.42143544464182,
        61.32206946477942,
        61.01966353728503,
        60.83308789104016,
    ];
    
    // Using default parameters: rsi_period=14, stoch_period=14, k=3, d=3
    const result = wasm.srsi_js(close, 14, 14, 3, 3);
    
    // Extract k and d from flattened result
    const k_values = result.slice(0, close.length);
    const d_values = result.slice(close.length);
    
    assert.strictEqual(k_values.length, close.length);
    assert.strictEqual(d_values.length, close.length);
    
    // Check last 5 values match expected
    const k_last5 = k_values.slice(-5);
    const d_last5 = d_values.slice(-5);
    
    assertArrayClose(k_last5, expected_k, 1e-6, "SRSI K last 5 values mismatch");
    assertArrayClose(d_last5, expected_d, 1e-6, "SRSI D last 5 values mismatch");
});

test('SRSI custom params', () => {
    // Test with custom parameters - mirrors check_srsi_custom_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 10, 10, 4, 4);
    assert.strictEqual(result.length, close.length * 2);
});

test('SRSI from slice', () => {
    // Test SRSI from slice data - mirrors check_srsi_from_slice
    const close = new Float64Array(testData.close);
    
    const result = wasm.srsi_js(close, 3, 3, 2, 2);
    assert.strictEqual(result.length, close.length * 2);
});

test('SRSI zero period', () => {
    const input_data = new Float64Array([10.0, 11.0, 12.0]);
    
    assert.throws(() => {
        wasm.srsi_js(input_data, 0, 0, 0, 0);
    }, /Invalid period/);
});

test('SRSI insufficient data', () => {
    const input_data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.srsi_js(input_data, 90, 3, 20, 20);
    }, /Not enough/);
});

test('SRSI empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.srsi_js(empty, 14, 14, 3, 3);
    }, /empty/);
});

test('SRSI fast API (srsi_into)', () => {
    const close = new Float64Array(testData.close);
    const k_out = new Float64Array(close.length);
    const d_out = new Float64Array(close.length);
    
    // Test normal operation (no aliasing)
    wasm.srsi_into(
        close.byteOffset, 
        k_out.byteOffset,
        d_out.byteOffset,
        close.length,
        14, 14, 3, 3
    );
    
    // Verify outputs are filled
    assert(!isNaN(k_out[50]));
    assert(!isNaN(d_out[50]));
});

test('SRSI fast API with aliasing', () => {
    const data = new Float64Array(testData.close);
    const data_copy = new Float64Array(data);
    
    // Test in-place operation (input aliased with k output)
    wasm.srsi_into(
        data.byteOffset,
        data.byteOffset,  // k_out same as input
        data_copy.byteOffset,  // d_out different
        data.length,
        14, 14, 3, 3
    );
    
    // The function should handle aliasing correctly
    assert(!isNaN(data[50]));  // k values
    assert(!isNaN(data_copy[50]));  // d values
});

test('SRSI batch operation', () => {
    const close = new Float64Array(testData.close.slice(0, 1000)); // Use smaller dataset
    
    const config = {
        rsi_period_range: [14, 14, 0],  // Default rsi_period only
        stoch_period_range: [14, 14, 0], // Default stoch_period only
        k_range: [3, 3, 0],  // Default k only
        d_range: [3, 3, 0]   // Default d only
    };
    
    const result = wasm.srsi_batch(close, config);
    
    assert(result.k_values);
    assert(result.d_values);
    assert(result.combos);
    assert.strictEqual(result.rows, 1); // 1 combination
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.k_values.length, close.length);
    assert.strictEqual(result.d_values.length, close.length);
});

test('SRSI batch with multiple params', () => {
    const close = new Float64Array(testData.close.slice(0, 500)); // Use smaller dataset
    
    const config = {
        rsi_period_range: [10, 14, 2],    // 10, 12, 14
        stoch_period_range: [10, 14, 2],  // 10, 12, 14
        k_range: [2, 4, 1],               // 2, 3, 4
        d_range: [2, 3, 1]                // 2, 3
    };
    
    const result = wasm.srsi_batch(close, config);
    
    // Should have 3 * 3 * 3 * 2 = 54 combinations
    const expected_rows = 3 * 3 * 3 * 2;
    assert.strictEqual(result.rows, expected_rows);
    assert.strictEqual(result.combos.length, expected_rows);
    assert.strictEqual(result.k_values.length, expected_rows * close.length);
    assert.strictEqual(result.d_values.length, expected_rows * close.length);
});

test('SRSI memory allocation and deallocation', () => {
    const len = 1000;
    
    // Allocate memory
    const ptr = wasm.srsi_alloc(len);
    assert(ptr !== 0, 'Allocation should return non-zero pointer');
    
    // Free memory
    assert.doesNotThrow(() => {
        wasm.srsi_free(ptr, len);
    });
});

test('SRSI batch fast API', () => {
    const close = new Float64Array(testData.close.slice(0, 500));
    
    // Calculate expected output size
    const rsi_periods = 3;    // (10, 14, 2) => 10, 12, 14
    const stoch_periods = 3;  // (10, 14, 2) => 10, 12, 14
    const k_periods = 3;      // (2, 4, 1) => 2, 3, 4
    const d_periods = 2;      // (2, 3, 1) => 2, 3
    const expected_rows = rsi_periods * stoch_periods * k_periods * d_periods;
    
    const k_out = new Float64Array(expected_rows * close.length);
    const d_out = new Float64Array(expected_rows * close.length);
    
    const rows = wasm.srsi_batch_into(
        close.byteOffset,
        k_out.byteOffset,
        d_out.byteOffset,
        close.length,
        10, 14, 2,    // rsi_period range
        10, 14, 2,    // stoch_period range
        2, 4, 1,      // k range
        2, 3, 1       // d range
    );
    
    assert.strictEqual(rows, expected_rows);
    
    // Verify outputs are filled
    assert(!isNaN(k_out[50]));
    assert(!isNaN(d_out[50]));
});