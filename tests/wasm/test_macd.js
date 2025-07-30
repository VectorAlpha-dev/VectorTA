/**
 * WASM binding tests for MACD indicator.
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

test('MACD partial params', () => {
    // Test with default parameters - mirrors check_macd_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    assert.strictEqual(result.values.length, close.length * 3); // 3 outputs
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
});

test('MACD accuracy', () => {
    // Test MACD matches expected values from Rust tests - mirrors check_macd_accuracy
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    
    // Extract individual arrays from flattened result
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    // Check last 5 values match expected from Rust tests
    const expected_macd = [
        -629.8674025082801,
        -600.2986584356258,
        -581.6188884820076,
        -551.1020443476082,
        -560.798510688488,
    ];
    const expected_signal = [
        -721.9744591891067,
        -697.6392990384105,
        -674.4352169271299,
        -649.7685824112256,
        -631.9745680666781,
    ];
    const expected_hist = [
        92.10705668082664,
        97.34064060278467,
        92.81632844512228,
        98.6665380636174,
        71.17605737819008,
    ];
    
    const last5_macd = macd.slice(-5);
    const last5_signal = signal.slice(-5);
    const last5_hist = hist.slice(-5);
    
    assertArrayClose(
        last5_macd,
        expected_macd,
        1e-1,
        "MACD last 5 values mismatch"
    );
    assertArrayClose(
        last5_signal,
        expected_signal,
        1e-1,
        "Signal last 5 values mismatch"
    );
    assertArrayClose(
        last5_hist,
        expected_hist,
        1e-1,
        "Histogram last 5 values mismatch"
    );
});

test('MACD zero period', () => {
    // Test MACD fails with zero period - mirrors check_macd_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.macd_js(inputData, 0, 26, 9, "ema");
    }, /Invalid period/);
});

test('MACD period exceeds length', () => {
    // Test MACD fails when period exceeds data length - mirrors check_macd_period_exceeds_length
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 12, 26, 9, "ema");
    }, /Invalid period/);
});

test('MACD very small dataset', () => {
    // Test MACD fails with insufficient data - mirrors check_macd_very_small_dataset
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 12, 26, 9, "ema");
    }, /Invalid period|Not enough valid data/);
});

test('MACD empty input', () => {
    // Test MACD fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.macd_js(empty, 12, 26, 9, "ema");
    }, /Input data slice is empty|Invalid period/);
});

test('MACD NaN handling', () => {
    // Test MACD handles NaN values correctly - mirrors check_macd_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    
    // Extract individual arrays
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    // After warmup period (approximately 240 values), there should be no NaNs
    if (close.length > 240) {
        for (let i = 240; i < close.length; i++) {
            assert(!isNaN(macd[i]), `Found unexpected NaN in MACD at index ${i}`);
            assert(!isNaN(signal[i]), `Found unexpected NaN in signal at index ${i}`);
            assert(!isNaN(hist[i]), `Found unexpected NaN in histogram at index ${i}`);
        }
    }
});

test('MACD all NaN input', () => {
    // Test MACD with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.macd_js(allNaN, 12, 26, 9, "ema");
    }, /All values are NaN/);
});

test('MACD fast API (in-place)', () => {
    // Test fast API with in-place operation
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate output buffers
    const macd_ptr = wasm.macd_alloc(len);
    const signal_ptr = wasm.macd_alloc(len);
    const hist_ptr = wasm.macd_alloc(len);
    
    try {
        // Compute MACD using fast API
        wasm.macd_into(
            close.byteOffset,
            macd_ptr,
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        // Read results from memory
        const memory = new Float64Array(wasm.memory.buffer);
        const macd = new Float64Array(memory.buffer, macd_ptr, len);
        const signal = new Float64Array(memory.buffer, signal_ptr, len);
        const hist = new Float64Array(memory.buffer, hist_ptr, len);
        
        // Compare with safe API
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        const safe_signal = safeResult.values.slice(len, len * 2);
        const safe_hist = safeResult.values.slice(len * 2);
        
        assertArrayClose(macd, safe_macd, 1e-10, "Fast API MACD mismatch");
        assertArrayClose(signal, safe_signal, 1e-10, "Fast API signal mismatch");
        assertArrayClose(hist, safe_hist, 1e-10, "Fast API histogram mismatch");
    } finally {
        // Clean up
        wasm.macd_free(macd_ptr, len);
        wasm.macd_free(signal_ptr, len);
        wasm.macd_free(hist_ptr, len);
    }
});

test('MACD fast API aliasing detection', () => {
    // Test that fast API handles aliasing correctly
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    // Allocate a single buffer to test aliasing
    const buffer_ptr = wasm.macd_alloc(len);
    const signal_ptr = wasm.macd_alloc(len);
    const hist_ptr = wasm.macd_alloc(len);
    
    try {
        // Copy input data to buffer
        const memory = new Float64Array(wasm.memory.buffer);
        const buffer = new Float64Array(memory.buffer, buffer_ptr, len);
        buffer.set(close);
        
        // Use same buffer as input and MACD output (aliasing)
        wasm.macd_into(
            buffer_ptr,
            buffer_ptr, // Same as input!
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        // Compare with safe API
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        
        assertArrayClose(buffer, safe_macd, 1e-10, "Aliasing handling failed");
    } finally {
        // Clean up
        wasm.macd_free(buffer_ptr, len);
        wasm.macd_free(signal_ptr, len);
        wasm.macd_free(hist_ptr, len);
    }
});

test('MACD batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [12, 12, 0],
        slow_period_range: [26, 26, 0],
        signal_period_range: [9, 9, 0],
        ma_type: "ema"
    });
    
    // Should have 1 row
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.macd.length, close.length);
    
    // Compare with single calculation
    const singleResult = wasm.macd_js(close, 12, 26, 9, "ema");
    const single_macd = singleResult.values.slice(0, close.length);
    
    assertArrayClose(
        batchResult.macd,
        single_macd,
        1e-10,
        "Batch vs single MACD mismatch"
    );
});

test('MACD batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50)); // Use smaller dataset for speed
    
    // Multiple parameter combinations
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [10, 14, 2],   // 10, 12, 14 (3 values)
        slow_period_range: [24, 28, 2],   // 24, 26, 28 (3 values)
        signal_period_range: [8, 10, 1],  // 8, 9, 10 (3 values)
        ma_type: "ema"
    });
    
    // Should have 3 * 3 * 3 = 27 combinations
    assert.strictEqual(batchResult.rows, 27);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.macd.length, 27 * 50);
    assert.strictEqual(batchResult.signal.length, 27 * 50);
    assert.strictEqual(batchResult.hist.length, 27 * 50);
    assert.strictEqual(batchResult.fast_periods.length, 27);
    assert.strictEqual(batchResult.slow_periods.length, 27);
    assert.strictEqual(batchResult.signal_periods.length, 27);
    
    // Check first combination (10, 24, 8)
    const firstMacd = batchResult.macd.slice(0, 50);
    const singleResult = wasm.macd_js(close, 10, 24, 8, "ema");
    const single_macd = singleResult.values.slice(0, 50);
    
    assertArrayClose(
        firstMacd,
        single_macd,
        1e-10,
        "First batch row should match single calculation"
    );
});

test('MACD unknown MA type', () => {
    // Test MACD fails with unknown moving average type
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 2, 3, 2, "unknown_ma");
    }, /Unknown MA type/);
});

test('MACD memory allocation/deallocation', () => {
    // Test memory management functions
    const len = 1000;
    
    // Allocate memory
    const ptr = wasm.macd_alloc(len);
    assert(ptr !== 0, "Allocation should return non-null pointer");
    
    // Free memory (should not throw)
    assert.doesNotThrow(() => {
        wasm.macd_free(ptr, len);
    });
    
    // Free null pointer (should not throw)
    assert.doesNotThrow(() => {
        wasm.macd_free(0, len);
    });
});

// Run Rust comparison if available
test.after(() => {
    if (process.env.RUN_RUST_COMPARISON) {
        compareWithRust('macd', testData);
    }
});