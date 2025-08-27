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
    
    // Use same tolerance as Rust tests (1e-1)
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
    
    // Allocate buffers
    const in_ptr = wasm.macd_alloc(len);
    const macd_ptr = wasm.macd_alloc(len);
    const signal_ptr = wasm.macd_alloc(len);
    const hist_ptr = wasm.macd_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const memory = wasm.__wbindgen_memory();
        const memView = new Float64Array(memory.buffer, in_ptr, len);
        memView.set(close);
        
        // Compute MACD using fast API
        // The function throws on error, so we don't need to check return value
        wasm.macd_into(
            in_ptr,
            macd_ptr,
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        // Read results from memory - recreate views after potential memory growth
        // Note: pointers are byte offsets, convert to element indices for subarray
        const memory2 = wasm.__wbindgen_memory();
        const fullView = new Float64Array(memory2.buffer);
        const macd = fullView.subarray(macd_ptr / 8, macd_ptr / 8 + len);
        const signal = fullView.subarray(signal_ptr / 8, signal_ptr / 8 + len);
        const hist = fullView.subarray(hist_ptr / 8, hist_ptr / 8 + len);
        
        // Store values before calling wasm.macd_js which might invalidate views
        const macdCopy = Array.from(macd);
        const signalCopy = Array.from(signal);
        const histCopy = Array.from(hist);
        
        // Compare with safe API
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        const safe_signal = safeResult.values.slice(len, len * 2);
        const safe_hist = safeResult.values.slice(len * 2);
        
        // Use the copied arrays for comparison since the original views were invalidated
        assertArrayClose(macdCopy, safe_macd, 1e-10, "Fast API MACD mismatch");
        assertArrayClose(signalCopy, safe_signal, 1e-10, "Fast API signal mismatch");
        assertArrayClose(histCopy, safe_hist, 1e-10, "Fast API histogram mismatch");
    } finally {
        // Clean up
        wasm.macd_free(in_ptr, len);
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
        const memory = wasm.__wbindgen_memory();
        const buffer = new Float64Array(memory.buffer, buffer_ptr, len);
        buffer.set(close);
        
        // Use same buffer as input and MACD output (aliasing)
        // The function handles aliasing internally and throws on error
        wasm.macd_into(
            buffer_ptr,
            buffer_ptr, // Same as input!
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        // Re-create view after potential memory changes
        // Use subarray for more reliable view creation
        const memory2 = wasm.__wbindgen_memory();
        const fullView = new Float64Array(memory2.buffer);
        const buffer_after = fullView.subarray(buffer_ptr / 8, buffer_ptr / 8 + len);
        
        // Store values before calling wasm.macd_js which might invalidate views
        const bufferCopy = Array.from(buffer_after);
        
        // Compare with safe API
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        
        assertArrayClose(bufferCopy, safe_macd, 1e-10, "Aliasing handling failed");
    } finally {
        // Clean up
        wasm.macd_free(buffer_ptr, len);
        wasm.macd_free(signal_ptr, len);
        wasm.macd_free(hist_ptr, len);
    }
});

test('MACD batch single parameter set', () => {
    // Test batch with single parameter combination using new flattened API
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [12, 12, 0],
        slow_period_range: [26, 26, 0],
        signal_period_range: [9, 9, 0],
        ma_type: "ema"
    });
    
    // Should have 1 row with flattened values
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.values.length, 3 * close.length); // 3 outputs flattened
    
    // Extract MACD from flattened format [macd block | signal block | hist block]
    const batch_macd = batchResult.values.slice(0, close.length);
    const batch_signal = batchResult.values.slice(close.length, 2 * close.length);
    const batch_hist = batchResult.values.slice(2 * close.length);
    
    // Compare with single calculation
    const singleResult = wasm.macd_js(close, 12, 26, 9, "ema");
    const single_macd = singleResult.values.slice(0, close.length);
    const single_signal = singleResult.values.slice(close.length, 2 * close.length);
    const single_hist = singleResult.values.slice(2 * close.length);
    
    assertArrayClose(
        batch_macd,
        single_macd,
        1e-10,
        "Batch vs single MACD mismatch"
    );
    assertArrayClose(
        batch_signal,
        single_signal,
        1e-10,
        "Batch vs single signal mismatch"
    );
    assertArrayClose(
        batch_hist,
        single_hist,
        1e-10,
        "Batch vs single histogram mismatch"
    );
});

test('MACD batch multiple parameters', () => {
    // Test batch with multiple parameter combinations using new flattened API
    const close = new Float64Array(testData.close.slice(0, 50)); // Use smaller dataset for speed
    
    // Multiple parameter combinations
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [10, 14, 2],   // 10, 12, 14 (3 values)
        slow_period_range: [24, 28, 2],   // 24, 26, 28 (3 values)
        signal_period_range: [8, 10, 1],  // 8, 9, 10 (3 values)
        ma_type: "ema"
    });
    
    // Should have 3 * 3 * 3 = 27 combinations with flattened output
    assert.strictEqual(batchResult.rows, 27);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 3 * 27 * 50); // 3 outputs * 27 rows * 50 cols
    assert.strictEqual(batchResult.fast_periods.length, 27);
    assert.strictEqual(batchResult.slow_periods.length, 27);
    assert.strictEqual(batchResult.signal_periods.length, 27);
    
    // Check first combination (10, 24, 8)
    // Values are arranged as [all macd rows | all signal rows | all hist rows]
    const macd_block_size = 27 * 50;
    const firstMacd = batchResult.values.slice(0, 50); // First row of MACD block
    
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

test('MACD warmup periods', () => {
    // Test MACD warmup periods match expected values
    const close = new Float64Array(testData.close);
    const fastPeriod = 12;
    const slowPeriod = 26;
    const signalPeriod = 9;
    
    const result = wasm.macd_js(close, fastPeriod, slowPeriod, signalPeriod, "ema");
    
    // Extract individual arrays
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    // MACD warmup: first slow_period - 1 values should be NaN
    const macdWarmup = slowPeriod - 1;
    for (let i = 0; i < macdWarmup; i++) {
        assert(isNaN(macd[i]), `Expected NaN at MACD index ${i} during warmup`);
    }
    
    // Signal and histogram warmup: first slow_period + signal_period - 2 values should be NaN
    const signalWarmup = slowPeriod + signalPeriod - 2;
    for (let i = 0; i < signalWarmup; i++) {
        assert(isNaN(signal[i]), `Expected NaN at signal index ${i} during warmup`);
        assert(isNaN(hist[i]), `Expected NaN at histogram index ${i} during warmup`);
    }
    
    // After warmup, values should not be NaN (for clean data)
    assert(!isNaN(macd[macdWarmup]), `Unexpected NaN at MACD index ${macdWarmup} after warmup`);
    assert(!isNaN(signal[signalWarmup]), `Unexpected NaN at signal index ${signalWarmup} after warmup`);
    assert(!isNaN(hist[signalWarmup]), `Unexpected NaN at histogram index ${signalWarmup} after warmup`);
});

test('MACD different MA types', () => {
    // Test MACD with different moving average types
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Test EMA (default)
    const resultEMA = wasm.macd_js(close, 12, 26, 9, "ema");
    const macdEMA = resultEMA.values.slice(0, close.length);
    assert.strictEqual(macdEMA.length, close.length);
    
    // Test SMA
    const resultSMA = wasm.macd_js(close, 12, 26, 9, "sma");
    const macdSMA = resultSMA.values.slice(0, close.length);
    assert.strictEqual(macdSMA.length, close.length);
    
    // Test WMA
    const resultWMA = wasm.macd_js(close, 12, 26, 9, "wma");
    const macdWMA = resultWMA.values.slice(0, close.length);
    assert.strictEqual(macdWMA.length, close.length);
    
    // Results should be different for different MA types (compare after warmup)
    let emaVsSmaMatch = true;
    let emaVsWmaMatch = true;
    for (let i = 50; i < close.length; i++) {
        if (Math.abs(macdEMA[i] - macdSMA[i]) > 1e-5) emaVsSmaMatch = false;
        if (Math.abs(macdEMA[i] - macdWMA[i]) > 1e-5) emaVsWmaMatch = false;
    }
    assert(!emaVsSmaMatch, "EMA and SMA should produce different results");
    assert(!emaVsWmaMatch, "EMA and WMA should produce different results");
});

test('MACD batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array(testData.close.slice(0, 50));
    
    // Test with step larger than range
    const largeBatch = wasm.macd_batch(close, {
        fast_period_range: [12, 14, 10], // Step 10 > range 2, should only get 12
        slow_period_range: [26, 26, 0],
        signal_period_range: [9, 9, 0],
        ma_type: "ema"
    });
    
    // Should only have 1 combination
    assert.strictEqual(largeBatch.rows, 1);
    assert.strictEqual(largeBatch.fast_periods.length, 1);
    assert.strictEqual(largeBatch.fast_periods[0], 12);
    
    // Test batch warmup periods
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [10, 14, 2],   // 3 values: 10, 12, 14
        slow_period_range: [24, 28, 2],   // 3 values: 24, 26, 28
        signal_period_range: [8, 8, 1],   // 1 value: 8
        ma_type: "ema"
    });
    
    // Should have 3 * 3 * 1 = 9 combinations
    assert.strictEqual(batchResult.rows, 9);
    
    // Check warmup periods for first combination (fast=10, slow=24, signal=8)
    // Values layout: [all macd rows | all signal rows | all hist rows]
    const firstRowMacd = batchResult.values.slice(0, 50);
    const firstRowSignal = batchResult.values.slice(9 * 50, 9 * 50 + 50);
    const firstRowHist = batchResult.values.slice(2 * 9 * 50, 2 * 9 * 50 + 50);
    
    // MACD warmup is slow - 1 = 24 - 1 = 23
    for (let i = 0; i < 23; i++) {
        assert(isNaN(firstRowMacd[i]), `Expected NaN in batch MACD at index ${i}`);
    }
    
    // Signal/hist warmup is slow + signal - 2 = 24 + 8 - 2 = 30
    for (let i = 0; i < 30; i++) {
        assert(isNaN(firstRowSignal[i]), `Expected NaN in batch signal at index ${i}`);
        assert(isNaN(firstRowHist[i]), `Expected NaN in batch histogram at index ${i}`);
    }
    
    // Test empty data
    assert.throws(() => {
        wasm.macd_batch(new Float64Array([]), {
            fast_period_range: [12, 12, 0],
            slow_period_range: [26, 26, 0],
            signal_period_range: [9, 9, 0],
            ma_type: "ema"
        });
    }, /All values are NaN|Input data slice is empty/);
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