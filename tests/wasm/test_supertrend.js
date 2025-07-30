/**
 * WASM binding tests for SuperTrend indicator.
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
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('SuperTrend partial params', () => {
    // Test with default parameters - mirrors check_supertrend_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Result is flattened: [trend..., changed...]
    const result = wasm.supertrend_js(high, low, close, 10, 3.0);
    assert.strictEqual(result.length, high.length * 2, 'Result should be 2x input length');
    
    // Extract trend and changed
    const trend = result.slice(0, high.length);
    const changed = result.slice(high.length);
    
    assert.strictEqual(trend.length, high.length);
    assert.strictEqual(changed.length, high.length);
});

test('SuperTrend accuracy', async () => {
    // Test SuperTrend matches expected values from Rust tests - mirrors check_supertrend_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.supertrend_js(high, low, close, 10, 3.0);
    
    // Extract trend and changed
    const trend = result.slice(0, high.length);
    const changed = result.slice(high.length);
    
    // Check last 5 values match expected from Rust tests
    const expectedLastFiveTrend = [
        61811.479454208165,
        61721.73150878735,
        61459.10835790861,
        61351.59752211775,
        61033.18776990598,
    ];
    const expectedLastFiveChanged = [0.0, 0.0, 0.0, 0.0, 0.0];
    
    assertArrayClose(
        trend.slice(-5),
        expectedLastFiveTrend,
        1e-4,
        "SuperTrend trend last 5 values mismatch"
    );
    
    assertArrayClose(
        changed.slice(-5),
        expectedLastFiveChanged,
        1e-9,
        "SuperTrend changed last 5 values mismatch"
    );
});

test('SuperTrend zero period', () => {
    // Test SuperTrend fails with zero period - mirrors check_supertrend_zero_period
    const high = new Float64Array([10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 11.0, 12.5]);
    const close = new Float64Array([9.5, 11.5, 13.0]);
    
    assert.throws(() => {
        wasm.supertrend_js(high, low, close, 0, 3.0);
    }, /Invalid period/);
});

test('SuperTrend period exceeds length', () => {
    // Test SuperTrend fails when period exceeds data length - mirrors check_supertrend_period_exceeds_length
    const high = new Float64Array([10.0, 12.0, 13.0]);
    const low = new Float64Array([9.0, 11.0, 12.5]);
    const close = new Float64Array([9.5, 11.5, 13.0]);
    
    assert.throws(() => {
        wasm.supertrend_js(high, low, close, 10, 3.0);
    }, /Invalid period/);
});

test('SuperTrend very small dataset', () => {
    // Test SuperTrend fails with dataset smaller than period - mirrors check_supertrend_very_small_dataset
    const high = new Float64Array([42.0]);
    const low = new Float64Array([40.0]);
    const close = new Float64Array([41.0]);
    
    assert.throws(() => {
        wasm.supertrend_js(high, low, close, 10, 3.0);
    }, /Invalid period|Not enough valid data/);
});

test('SuperTrend empty input', () => {
    // Test SuperTrend fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.supertrend_js(empty, empty, empty, 10, 3.0);
    }, /Empty data/);
});

test('SuperTrend all NaN', () => {
    // Test SuperTrend with all NaN values
    const size = 100;
    const high = new Float64Array(size).fill(NaN);
    const low = new Float64Array(size).fill(NaN);
    const close = new Float64Array(size).fill(NaN);
    
    assert.throws(() => {
        wasm.supertrend_js(high, low, close, 10, 3.0);
    }, /All values are NaN/);
});

test('SuperTrend NaN handling', () => {
    // Test SuperTrend handles NaN values correctly - mirrors check_supertrend_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // Insert some NaN values
    for (let i = 0; i < 5; i++) {
        high[i] = NaN;
        low[i] = NaN;
        close[i] = NaN;
    }
    
    const result = wasm.supertrend_js(high, low, close, 10, 3.0);
    const trend = result.slice(0, high.length);
    const changed = result.slice(high.length);
    
    assert.strictEqual(trend.length, high.length);
    assert.strictEqual(changed.length, high.length);
    
    // First few values should be NaN
    for (let i = 0; i < 5; i++) {
        assert(isNaN(trend[i]), `Expected NaN at index ${i}`);
    }
});

test('SuperTrend reinput', () => {
    // Test SuperTrend applied to its own output - mirrors check_supertrend_reinput
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    // First pass
    const result1 = wasm.supertrend_js(high, low, close, 10, 3.0);
    const trend1 = result1.slice(0, high.length);
    
    // Second pass - apply SuperTrend to trend output
    const result2 = wasm.supertrend_js(trend1, trend1, trend1, 5, 2.0);
    const trend2 = result2.slice(0, trend1.length);
    const changed2 = result2.slice(trend1.length);
    
    assert.strictEqual(trend2.length, trend1.length);
    assert.strictEqual(changed2.length, trend1.length);
});

test('SuperTrend fast API (no aliasing)', () => {
    const size = 20;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    // Generate test data
    for (let i = 0; i < size; i++) {
        close[i] = 50 + i * 0.5 + Math.sin(i) * 2;
        high[i] = close[i] + Math.random() * 2;
        low[i] = close[i] - Math.random() * 2;
    }
    
    // Allocate memory
    const highPtr = wasm.supertrend_alloc(size);
    const lowPtr = wasm.supertrend_alloc(size);
    const closePtr = wasm.supertrend_alloc(size);
    const trendPtr = wasm.supertrend_alloc(size);
    const changedPtr = wasm.supertrend_alloc(size);
    
    try {
        // Copy input data to WASM memory
        const memory = new Float64Array(wasm.memory.buffer);
        memory.set(high, highPtr / 8);
        memory.set(low, lowPtr / 8);
        memory.set(close, closePtr / 8);
        
        // Call fast API
        wasm.supertrend_into(
            highPtr, lowPtr, closePtr,
            trendPtr, changedPtr,
            size, 5, 2.0
        );
        
        // Read results
        const trend = Array.from(memory.slice(trendPtr / 8, trendPtr / 8 + size));
        const changed = Array.from(memory.slice(changedPtr / 8, changedPtr / 8 + size));
        
        // Verify results
        assert.strictEqual(trend.length, size);
        assert.strictEqual(changed.length, size);
        
        // Changed values should be 0 or 1
        for (let i = 0; i < changed.length; i++) {
            if (!isNaN(changed[i])) {
                assert(changed[i] === 0.0 || changed[i] === 1.0, 
                    `Changed value at ${i} should be 0 or 1: ${changed[i]}`);
            }
        }
        
        // Compare with safe API
        const safeResult = wasm.supertrend_js(high, low, close, 5, 2.0);
        const safeTrend = safeResult.slice(0, size);
        const safeChanged = safeResult.slice(size);
        
        assertArrayClose(trend, safeTrend, 1e-10, "Fast vs Safe trend mismatch");
        assertArrayClose(changed, safeChanged, 1e-10, "Fast vs Safe changed mismatch");
        
    } finally {
        // Free memory
        wasm.supertrend_free(highPtr, size);
        wasm.supertrend_free(lowPtr, size);
        wasm.supertrend_free(closePtr, size);
        wasm.supertrend_free(trendPtr, size);
        wasm.supertrend_free(changedPtr, size);
    }
});

test('SuperTrend fast API (with aliasing)', () => {
    const size = 20;
    const data = new Float64Array(size);
    
    // Generate test data
    for (let i = 0; i < size; i++) {
        data[i] = 50 + i * 0.5;
    }
    
    // Allocate memory
    const dataPtr = wasm.supertrend_alloc(size);
    const trendPtr = dataPtr; // Aliasing: trend output overwrites high input
    const changedPtr = wasm.supertrend_alloc(size);
    
    try {
        // Copy input data to WASM memory
        const memory = new Float64Array(wasm.memory.buffer);
        memory.set(data, dataPtr / 8);
        
        // Call fast API with aliasing
        wasm.supertrend_into(
            dataPtr, dataPtr, dataPtr,  // All inputs from same location
            trendPtr, changedPtr,       // Trend aliases with input
            size, 5, 2.0
        );
        
        // Read results
        const trend = Array.from(memory.slice(trendPtr / 8, trendPtr / 8 + size));
        const changed = Array.from(memory.slice(changedPtr / 8, changedPtr / 8 + size));
        
        // Verify results
        assert.strictEqual(trend.length, size);
        assert.strictEqual(changed.length, size);
        
        // Results should be valid despite aliasing
        let hasValidValues = false;
        for (let i = 5; i < trend.length; i++) {
            if (!isNaN(trend[i])) {
                hasValidValues = true;
                break;
            }
        }
        assert(hasValidValues, 'Should have valid trend values after warmup');
        
    } finally {
        // Free memory
        wasm.supertrend_free(dataPtr, size);
        wasm.supertrend_free(changedPtr, size);
    }
});

test('SuperTrend batch API', () => {
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [8, 12, 2],    // 3 values: 8, 10, 12
        factor_range: [2.0, 3.0, 0.5] // 3 values: 2.0, 2.5, 3.0
    };
    
    const result = wasm.supertrend_batch(high, low, close, config);
    
    // Should have 3 * 3 = 9 combinations
    assert.strictEqual(result.rows, 9, 'Should have 9 parameter combinations');
    assert.strictEqual(result.cols, 100, 'Should have same length as input');
    assert.strictEqual(result.trend.length, 900, 'Trend should have rows*cols values');
    assert.strictEqual(result.changed.length, 900, 'Changed should have rows*cols values');
    assert.strictEqual(result.periods.length, 9, 'Should have 9 period values');
    assert.strictEqual(result.factors.length, 9, 'Should have 9 factor values');
    
    // Verify parameter combinations
    const expectedPeriods = [8, 8, 8, 10, 10, 10, 12, 12, 12];
    const expectedFactors = [2.0, 2.5, 3.0, 2.0, 2.5, 3.0, 2.0, 2.5, 3.0];
    
    assertArrayClose(result.periods, expectedPeriods, 1e-10, "Batch periods mismatch");
    assertArrayClose(result.factors, expectedFactors, 1e-10, "Batch factors mismatch");
    
    // Verify one specific combination matches single calculation
    const singleResult = wasm.supertrend_js(high, low, close, 10, 2.5);
    const singleTrend = singleResult.slice(0, 100);
    
    // Row 4 should be period=10, factor=2.5 (index 4 in flattened array)
    const batchTrend = result.trend.slice(4 * 100, 5 * 100);
    assertArrayClose(batchTrend, singleTrend, 1e-10, "Batch vs single trend mismatch");
});

test('SuperTrend edge cases', () => {
    // Test with very small factor
    const high = new Float64Array([100.0, 101.0, 102.0, 101.5, 100.5]);
    const low = new Float64Array([99.0, 100.0, 101.0, 100.5, 99.5]);
    const close = new Float64Array([99.5, 100.5, 101.5, 101.0, 100.0]);
    
    const result1 = wasm.supertrend_js(high, low, close, 2, 0.1);
    const trend1 = result1.slice(0, high.length);
    const changed1 = result1.slice(high.length);
    
    assert.strictEqual(trend1.length, high.length);
    assert.strictEqual(changed1.length, high.length);
    
    // Test with large factor
    const result2 = wasm.supertrend_js(high, low, close, 2, 10.0);
    const trend2 = result2.slice(0, high.length);
    const changed2 = result2.slice(high.length);
    
    assert.strictEqual(trend2.length, high.length);
    assert.strictEqual(changed2.length, high.length);
});