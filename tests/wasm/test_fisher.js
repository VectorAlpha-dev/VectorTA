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
    
    // Fisher returns flattened array [fisher..., signal...]
    const result = wasm.fisher_js(high, low, 9);
    assert.strictEqual(result.length, high.length * 2);
    
    // Extract fisher and signal arrays
    const fisher = result.slice(0, high.length);
    const signal = result.slice(high.length);
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
});

test('Fisher accuracy', () => {
    // Test Fisher matches expected values from Rust tests - mirrors check_fisher_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    
    // Extract fisher and signal arrays
    const fisher = result.slice(0, high.length);
    const signal = result.slice(high.length);
    
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
    const fisher1 = new Float64Array(result1.slice(0, high.length));
    const signal1 = new Float64Array(result1.slice(high.length));
    
    assert.strictEqual(fisher1.length, high.length);
    assert.strictEqual(signal1.length, high.length);
    
    // Second pass - use fisher as high and signal as low
    const result2 = wasm.fisher_js(fisher1, signal1, 3);
    const fisher2 = result2.slice(0, fisher1.length);
    const signal2 = result2.slice(fisher1.length);
    
    assert.strictEqual(fisher2.length, fisher1.length);
    assert.strictEqual(signal2.length, signal1.length);
});

test('Fisher nan handling', () => {
    // Test Fisher handles NaN values correctly - mirrors check_fisher_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    const fisher = result.slice(0, high.length);
    const signal = result.slice(high.length);
    
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
    
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
    
    // Allocate output arrays
    const fisherOut = new Float64Array(high.length);
    const signalOut = new Float64Array(high.length);
    
    // Call fast API
    wasm.fisher_into(
        high.buffer,
        low.buffer,
        fisherOut.buffer,
        signalOut.buffer,
        high.length,
        3
    );
    
    // Verify results match safe API
    const safeResult = wasm.fisher_js(high, low, 3);
    const safeFisher = safeResult.slice(0, high.length);
    const safeSignal = safeResult.slice(high.length);
    
    assertArrayClose(fisherOut, safeFisher, 1e-10, "Fast API fisher mismatch");
    assertArrayClose(signalOut, safeSignal, 1e-10, "Fast API signal mismatch");
});

test('Fisher fast API aliasing', () => {
    // Test fast API with aliasing (in-place operation)
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    
    // Copy for comparison
    const highCopy = new Float64Array(high);
    const lowCopy = new Float64Array(low);
    
    // Call fast API with aliasing (fisher output overwrites high)
    wasm.fisher_into(
        high.buffer,
        low.buffer,
        high.buffer,  // Output to same buffer as input
        low.buffer,   // Output to same buffer as input
        high.length,
        3
    );
    
    // Verify results match safe API
    const safeResult = wasm.fisher_js(highCopy, lowCopy, 3);
    const safeFisher = safeResult.slice(0, highCopy.length);
    const safeSignal = safeResult.slice(highCopy.length);
    
    assertArrayClose(high, safeFisher, 1e-10, "Fast API aliased fisher mismatch");
    assertArrayClose(low, safeSignal, 1e-10, "Fast API aliased signal mismatch");
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
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 1 * 100 * 2);  // rows * cols * 2 outputs
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.combos[0].period, 9);
    
    // Extract first row fisher and signal
    const fisher = result.values.slice(0, 100);
    const signal = result.values.slice(100, 200);
    
    // Compare with single calculation
    const singleResult = wasm.fisher_js(high, low, 9);
    const singleFisher = singleResult.slice(0, 100);
    const singleSignal = singleResult.slice(100);
    
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
    
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 3 * 50 * 2);  // rows * cols * 2 outputs
    assert.strictEqual(result.combos.length, 3);
    
    const periods = result.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [5, 7, 9]);
    
    // Verify each row matches individual calculation
    for (let i = 0; i < 3; i++) {
        const period = periods[i];
        const rowStart = i * 50 * 2;
        const fisher = result.values.slice(rowStart, rowStart + 50);
        const signal = result.values.slice(rowStart + 50, rowStart + 100);
        
        const singleResult = wasm.fisher_js(high, low, period);
        const singleFisher = singleResult.slice(0, 50);
        const singleSignal = singleResult.slice(50);
        
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