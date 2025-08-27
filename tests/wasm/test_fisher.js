/**
 * WASM binding tests for Fisher Transform indicator.
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

test('Fisher partial params', () => {
    // Test with default parameters - mirrors check_fisher_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Fisher returns FisherResult with values, rows, cols
    const result = wasm.fisher_js(high, low, 9);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.values.length, high.length * 2);
    
    // Extract fisher and signal arrays
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
});

test('Fisher accuracy', () => {
    // Test Fisher matches expected values from Rust tests - mirrors check_fisher_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    
    // Extract fisher and signal arrays from FisherResult
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
    
    // Expected last 5 values from Rust tests
    const expectedLast5Fisher = [
        -0.4720164683904261,
        -0.23467530106650444,
        -0.14879388501136784,
        -0.026651419122953053,
        -0.2569225042442664,
    ];
    
    // Check last 5 values match expected (with looser tolerance for Fisher)
    const last5 = fisher.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5Fisher,
        0.1,  // 10% tolerance as in Rust tests
        "Fisher last 5 values mismatch"
    );
});

test('Fisher zero period', () => {
    // Test Fisher fails with zero period - mirrors check_fisher_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 0);
    }, /Invalid period/);
});

test('Fisher period exceeds length', () => {
    // Test Fisher fails when period exceeds data length - mirrors check_fisher_period_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 10);
    }, /Invalid period/);
});

test('Fisher very small dataset', () => {
    // Test Fisher fails with insufficient data - mirrors check_fisher_very_small_dataset
    const high = new Float64Array([10.0]);
    const low = new Float64Array([5.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 9);
    }, /Invalid period|Not enough valid data/);
});

test('Fisher empty input', () => {
    // Test Fisher fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.fisher_js(empty, empty, 9);
    }, /Empty data/);
});

test('Fisher reinput', () => {
    // Test Fisher applied to Fisher output - mirrors check_fisher_reinput
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    
    // First pass
    const result1 = wasm.fisher_js(high, low, 3);
    const fisher1 = new Float64Array(result1.values.slice(0, high.length));
    const signal1 = new Float64Array(result1.values.slice(high.length));
    
    assert.strictEqual(fisher1.length, high.length);
    assert.strictEqual(signal1.length, high.length);
    
    // Second pass - use fisher as high and signal as low
    const result2 = wasm.fisher_js(fisher1, signal1, 3);
    const fisher2 = result2.values.slice(0, fisher1.length);
    const signal2 = result2.values.slice(fisher1.length);
    
    assert.strictEqual(fisher2.length, fisher1.length);
    assert.strictEqual(signal2.length, signal1.length);
});

test('Fisher nan handling', () => {
    // Test Fisher handles NaN values correctly - mirrors check_fisher_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
    
    // First period-1 values should be NaN (warmup period)
    assertAllNaN(fisher.slice(0, 8), "Expected NaN in warmup period for fisher");
    assertAllNaN(signal.slice(0, 8), "Expected NaN in warmup period for signal");
    
    // After warmup period, no NaN values should exist
    if (fisher.length > 240) {
        assertNoNaN(fisher.slice(240), "Found NaN after warmup in fisher");
        assertNoNaN(signal.slice(240), "Found NaN after warmup in signal");
    }
});

test('Fisher all nan input', () => {
    // Test Fisher fails with all NaN values
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.fisher_js(allNan, allNan, 9);
    }, /All values are NaN/);
});

// Fast API tests
test('Fisher fast API basic', () => {
    // Test fast API basic functionality
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    const len = high.length;
    
    // Allocate memory in WASM
    const highPtr = wasm.fisher_alloc(len);
    const lowPtr = wasm.fisher_alloc(len);
    const outPtr = wasm.fisher_alloc(len * 2);  // Single output buffer for both
    
    try {
        // Create views into WASM memory
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        
        // Copy input data to WASM memory
        highView.set(high);
        lowView.set(low);
        
        // Call fast API with new signature
        wasm.fisher_into(
            highPtr,
            lowPtr,
            outPtr,  // Single output pointer
            len,
            3
        );
        
        // Read results from WASM memory (recreate views in case memory grew)
        const output = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * 2);
        const fisherOut = output.slice(0, len);
        const signalOut = output.slice(len);
        
        // Verify results match safe API
        const safeResult = wasm.fisher_js(high, low, 3);
        const safeFisher = safeResult.values.slice(0, len);
        const safeSignal = safeResult.values.slice(len);
        
        assertArrayClose(fisherOut, safeFisher, 1e-10, "Fast API fisher mismatch");
        assertArrayClose(signalOut, safeSignal, 1e-10, "Fast API signal mismatch");
    } finally {
        // Clean up allocated memory
        wasm.fisher_free(highPtr, len);
        wasm.fisher_free(lowPtr, len);
        wasm.fisher_free(outPtr, len * 2);
    }
});

test('Fisher fast API aliasing', () => {
    // Test fast API with aliasing (in-place operation)
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    const len = high.length;
    
    // Copy for comparison
    const highCopy = new Float64Array(high);
    const lowCopy = new Float64Array(low);
    
    // Allocate memory in WASM for combined input/output
    const dataPtr = wasm.fisher_alloc(len * 2);
    
    try {
        // Create views and copy input data (high in first half, low in second half)
        let dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len * 2);
        dataView.set(high, 0);
        dataView.set(low, len);
        
        // Call fast API with aliasing (output overwrites input)
        wasm.fisher_into(
            dataPtr,       // high pointer
            dataPtr + len * 8,  // low pointer (offset by len * sizeof(f64))
            dataPtr,       // Output to same buffer as input
            len,
            3
        );
        
        // Read results (recreate views in case memory grew)
        dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len * 2);
        const fisherOut = dataView.slice(0, len);
        const signalOut = dataView.slice(len);
        
        // Verify results match safe API
        const safeResult = wasm.fisher_js(highCopy, lowCopy, 3);
        const safeFisher = safeResult.values.slice(0, len);
        const safeSignal = safeResult.values.slice(len);
        
        assertArrayClose(fisherOut, safeFisher, 1e-10, "Fast API aliased fisher mismatch");
        assertArrayClose(signalOut, safeSignal, 1e-10, "Fast API aliased signal mismatch");
    } finally {
        // Clean up allocated memory
        wasm.fisher_free(dataPtr, len * 2);
    }
});

// Batch API tests
test('Fisher batch single period', () => {
    // Test batch API with single period
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [9, 9, 1]
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    assert.strictEqual(result.rows, 2);  // 2 * combos.len() for interleaved fisher/signal
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 2 * 100);  // rows * cols
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.combos[0].period, 9);
    
    // Extract first combo's fisher and signal (interleaved)
    const fisher = result.values.slice(0, 100);
    const signal = result.values.slice(100, 200);
    
    // Compare with single calculation
    const singleResult = wasm.fisher_js(high, low, 9);
    const singleFisher = singleResult.values.slice(0, 100);
    const singleSignal = singleResult.values.slice(100);
    
    assertArrayClose(fisher, singleFisher, 1e-10, "Batch fisher mismatch");
    assertArrayClose(signal, singleSignal, 1e-10, "Batch signal mismatch");
});

test('Fisher batch multiple periods', () => {
    // Test batch API with multiple periods
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [5, 9, 2]  // periods: 5, 7, 9
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    assert.strictEqual(result.rows, 6);  // 2 * 3 combos for interleaved fisher/signal
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);  // rows * cols
    assert.strictEqual(result.combos.length, 3);
    
    const periods = result.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [5, 7, 9]);
    
    // Verify each combo matches individual calculation (interleaved layout)
    for (let i = 0; i < 3; i++) {
        const period = periods[i];
        const rowStart = i * 2 * 50;  // Each combo has 2 rows (fisher, signal)
        const fisher = result.values.slice(rowStart, rowStart + 50);
        const signal = result.values.slice(rowStart + 50, rowStart + 100);
        
        const singleResult = wasm.fisher_js(high, low, period);
        const singleFisher = singleResult.values.slice(0, 50);
        const singleSignal = singleResult.values.slice(50);
        
        assertArrayClose(fisher, singleFisher, 1e-10, `Batch fisher period ${period} mismatch`);
        assertArrayClose(signal, singleSignal, 1e-10, `Batch signal period ${period} mismatch`);
    }
});

// Memory management tests
test('Fisher memory allocation and deallocation', () => {
    // Test memory management functions
    const len = 1000;
    
    // Allocate memory
    const ptr = wasm.fisher_alloc(len);
    assert(ptr !== 0, "Failed to allocate memory");
    
    // Free memory
    assert.doesNotThrow(() => {
        wasm.fisher_free(ptr, len);
    });
    
    // Test null pointer safety
    assert.doesNotThrow(() => {
        wasm.fisher_free(0, len);
    });
});

// Additional tests for missing coverage
test('Fisher mismatched lengths', () => {
    // Test Fisher fails with mismatched input lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);  // Different length
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 2);
    }, /Mismatched data length/);
});

test('Fisher warmup period behavior', () => {
    // Test Fisher warmup period behavior matches expectations
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const period = 9;
    
    const result = wasm.fisher_js(high, low, period);
    const fisher = result.values.slice(0, 50);
    const signal = result.values.slice(50);
    
    // Find first non-NaN value
    let firstValid = -1;
    for (let i = 0; i < fisher.length; i++) {
        if (!isNaN(fisher[i])) {
            firstValid = i;
            break;
        }
    }
    
    // Should be at index period-1 (8 for period=9)
    assert.strictEqual(firstValid, period - 1, `First valid value at wrong index: ${firstValid} vs expected ${period-1}`);
    
    // Verify signal lag property: signal[i] should equal fisher[i-1] for i > warmup
    for (let i = period; i < fisher.length; i++) {
        assertClose(signal[i], fisher[i-1], 1e-10, `Signal lag property violated at index ${i}`);
    }
});

test('Fisher extreme values', () => {
    // Test Fisher with extreme values
    const high = new Float64Array([1e10, 1e11, 1e12, 1e13, 1e14]);
    const low = new Float64Array([1e9, 1e10, 1e11, 1e12, 1e13]);
    
    const result = wasm.fisher_js(high, low, 3);
    const fisher = result.values.slice(0, 5);
    const signal = result.values.slice(5);
    
    // Should not produce inf or nan after warmup
    for (let i = 2; i < 5; i++) {
        assert(isFinite(fisher[i]), `Fisher produced non-finite value at index ${i}`);
        assert(isFinite(signal[i]), `Signal produced non-finite value at index ${i}`);
    }
});

test('Fisher constant price', () => {
    // Test Fisher behavior with constant prices
    const high = new Float64Array(20).fill(100.0);
    const low = new Float64Array(20).fill(100.0);
    
    const result = wasm.fisher_js(high, low, 5);
    const fisher = result.values.slice(0, 20);
    
    // With constant prices (high == low), Fisher transform will hit division
    // protection and produce specific values, not necessarily zero
    // Just check that values are finite and defined after warmup
    for (let i = 4; i < 20; i++) {  // warmup is period-1 = 4
        assert(!isNaN(fisher[i]), `Fisher should not be NaN at index ${i}`);
        assert(isFinite(fisher[i]), `Fisher should be finite at index ${i}`);
    }
});

test('Fisher batch comprehensive sweep', () => {
    // Test batch with comprehensive parameter sweep
    const high = new Float64Array(testData.high.slice(0, 30));
    const low = new Float64Array(testData.low.slice(0, 30));
    
    const config = {
        period_range: [3, 9, 3]  // periods: 3, 6, 9
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    // Should have 3 combos, so 6 rows for interleaved fisher/signal
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.combos.length, 3);
    
    const periods = [3, 6, 9];
    
    // Verify warmup periods are correct for each combo
    for (let i = 0; i < 3; i++) {
        const period = periods[i];
        const rowStart = i * 2 * 30;
        const fisher = result.values.slice(rowStart, rowStart + 30);
        const signal = result.values.slice(rowStart + 30, rowStart + 60);
        
        // Check warmup NaNs
        for (let j = 0; j < period - 1; j++) {
            assert(isNaN(fisher[j]), `Expected NaN at index ${j} for period ${period}`);
            assert(isNaN(signal[j]), `Expected NaN at index ${j} for period ${period}`);
        }
        
        // First valid should be at period-1
        assert(!isNaN(fisher[period - 1]), `Expected valid value at index ${period - 1} for period ${period}`);
    }
});