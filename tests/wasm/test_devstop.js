/**
 * WASM binding tests for DevStop indicator.
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

test('DevStop partial params', () => {
    // Test with default parameters - mirrors check_devstop_partial_params
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);
    
    // Test with custom params
    const resultCustom = wasm.devstop(high, low, 20, 1.0, 2, 'short', 'ema');
    assert.strictEqual(resultCustom.length, high.length);
});

test('DevStop accuracy', async () => {
    // Test DevStop matches expected values from Rust tests - mirrors check_devstop_accuracy
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.devstop;
    
    const result = wasm.devstop(
        high, low,
        expected.defaultParams.period,
        expected.defaultParams.mult,
        expected.defaultParams.devtype,
        expected.defaultParams.direction,
        expected.defaultParams.maType
    );
    
    assert.strictEqual(result.length, high.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "DevStop last 5 values mismatch"
    );
    
    // Compare full output with Rust
    const params = {
        high: 'high',
        low: 'low',
        ...expected.defaultParams
    };
    await compareWithRust('devstop', result, null, params);
});

test('DevStop default candles', () => {
    // Test DevStop with default parameters - mirrors check_devstop_default_candles
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);
});

test('DevStop zero period', () => {
    // Test DevStop fails with zero period - mirrors check_devstop_zero_period
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.devstop(high, low, 0, 1.0, 0, 'long', 'sma');
    }, /Invalid period/);
});

test('DevStop period exceeds length', () => {
    // Test DevStop fails when period exceeds data length - mirrors check_devstop_period_exceeds_length
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.devstop(high, low, 10, 1.0, 0, 'long', 'sma');
    }, /Invalid period/);
});

test('DevStop very small dataset', () => {
    // Test DevStop fails with insufficient data - mirrors check_devstop_very_small_dataset
    const high = new Float64Array([100.0]);
    const low = new Float64Array([90.0]);
    
    assert.throws(() => {
        wasm.devstop(high, low, 20, 2.0, 0, 'long', 'sma');
    }, /Invalid period|Not enough valid data/);
});

test('DevStop NaN handling', () => {
    // Test DevStop handles NaN values correctly - mirrors check_devstop_nan_handling
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    assert.strictEqual(result.length, high.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // Warmup period for devstop: first + 2*period - 1
    // With first=0 and period=20: 0 + 2*20 - 1 = 39
    const expectedWarmup = 39;
    if (result.length > expectedWarmup) {
        // Check that warmup period has some NaN values
        let hasNaN = false;
        for (let i = 0; i < expectedWarmup; i++) {
            if (isNaN(result[i])) {
                hasNaN = true;
                break;
            }
        }
        assert(hasNaN, "Expected NaN in warmup period");
    }
});

test('DevStop all NaN input', () => {
    // Test DevStop with all NaN values
    const allNaNHigh = new Float64Array(100);
    const allNaNLow = new Float64Array(100);
    allNaNHigh.fill(NaN);
    allNaNLow.fill(NaN);
    
    assert.throws(() => {
        wasm.devstop(allNaNHigh, allNaNLow, 20, 0.0, 0, 'long', 'sma');
    }, /All values are NaN/);
});

test('DevStop mismatched lengths', () => {
    // Test DevStop with mismatched high/low lengths
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(() => {
        wasm.devstop(high, low, 2, 0.0, 0, 'long', 'sma');
    }, /length mismatch/);
});

test('DevStop batch single parameter set', () => {
    // Test batch with single parameter combination
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    // Using the new ergonomic batch API for single parameter
    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [20, 20, 0],
        mult_range: [0.0, 0.0, 0],
        devtype_range: [0, 0, 0]
    });
    
    // Should match single calculation
    const singleResult = wasm.devstop(high, low, 20, 0.0, 0, 'long', 'sma');
    
    assert.strictEqual(batchResult.values.length, singleResult.length);
    // Allow a tiny absolute tolerance to account for benign path differences
    // between batch and single implementations (still far stricter than Rust ref checks)
    assertArrayClose(batchResult.values, singleResult, 1e-9, "Batch vs single mismatch");
});

test('DevStop batch multiple periods', () => {
    // Test batch with multiple period values
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset for speed
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Multiple periods: 10, 15, 20 using ergonomic API
    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [10, 20, 5],      // period range
        mult_range: [0.0, 0.0, 0],      // mult range  
        devtype_range: [0, 0, 0]        // devtype range
    });
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.values.length, 3 * 100);
    assert.strictEqual(batchResult.rows, 3);
    assert.strictEqual(batchResult.cols, 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.values.slice(rowStart, rowEnd);
        
        const singleResult = wasm.devstop(high, low, periods[i], 0.0, 0, 'long', 'sma');
        // Use a very small absolute tolerance for equality between paths
        assertArrayClose(
            Array.from(rowData),
            Array.from(singleResult),
            1e-9,
            `Batch row ${i} (period=${periods[i]}) mismatch`
        );
    }
});

test('DevStop direction types', () => {
    // Test DevStop with different direction types
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Test long direction
    const resultLong = wasm.devstop(high, low, 20, 1.0, 0, 'long', 'sma');
    assert.strictEqual(resultLong.length, high.length);
    
    // Test short direction
    const resultShort = wasm.devstop(high, low, 20, 1.0, 0, 'short', 'sma');
    assert.strictEqual(resultShort.length, high.length);
    
    // Results should be different for long vs short
    let isDifferent = false;
    for (let i = 0; i < resultLong.length; i++) {
        if (!isNaN(resultLong[i]) && !isNaN(resultShort[i])) {
            if (Math.abs(resultLong[i] - resultShort[i]) > 1e-10) {
                isDifferent = true;
                break;
            }
        }
    }
    assert(isDifferent, "Long and short directions should produce different results");
});

test('DevStop MA types', () => {
    // Test DevStop with different MA types
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const maTypes = ['sma', 'ema', 'wma', 'hma', 'dema'];
    const results = {};
    
    for (const maType of maTypes) {
        results[maType] = wasm.devstop(high, low, 20, 1.0, 0, 'long', maType);
        assert.strictEqual(results[maType].length, high.length, `MA type ${maType} length mismatch`);
    }
    
    // Different MA types should produce different results
    for (let i = 0; i < maTypes.length - 1; i++) {
        const ma1 = maTypes[i];
        for (let j = i + 1; j < maTypes.length; j++) {
            const ma2 = maTypes[j];
            let isDifferent = false;
            for (let k = 0; k < results[ma1].length; k++) {
                if (!isNaN(results[ma1][k]) && !isNaN(results[ma2][k])) {
                    if (Math.abs(results[ma1][k] - results[ma2][k]) > 1e-10) {
                        isDifferent = true;
                        break;
                    }
                }
            }
            assert(isDifferent, `${ma1} and ${ma2} should produce different results`);
        }
    }
});

test('DevStop devtype variations', () => {
    // Test DevStop with different deviation types
    const high = new Float64Array(testData.high.slice(0, 100)); // Use smaller dataset
    const low = new Float64Array(testData.low.slice(0, 100));
    
    // Test all three deviation types
    const results = {};
    for (const devtype of [0, 1, 2]) {
        results[devtype] = wasm.devstop(high, low, 20, 1.0, devtype, 'long', 'sma');
        assert.strictEqual(results[devtype].length, high.length, `Devtype ${devtype} length mismatch`);
    }
    
    // Different deviation types should produce different results when mult > 0
    for (const [dt1, dt2] of [[0, 1], [0, 2], [1, 2]]) {
        let isDifferent = false;
        for (let i = 0; i < results[dt1].length; i++) {
            if (!isNaN(results[dt1][i]) && !isNaN(results[dt2][i])) {
                if (Math.abs(results[dt1][i] - results[dt2][i]) > 1e-10) {
                    isDifferent = true;
                    break;
                }
            }
        }
        assert(isDifferent, `Devtype ${dt1} and ${dt2} should produce different results`);
    }
});

test('DevStop batch parameter sweep', () => {
    // Test batch with comprehensive parameter sweep
    // Need at least 60 data points for period=30 (warmup = 59)
    const high = new Float64Array(testData.high.slice(0, 100)); // Use 100 for sufficient data
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const batchResult = wasm.devstop_batch(high, low, {
        period_range: [10, 30, 10],    // 10, 20, 30 (3 values)
        mult_range: [0.0, 1.0, 0.5],   // 0.0, 0.5, 1.0 (3 values)
        devtype_range: [0, 2, 1]       // 0, 1, 2 (3 values)
    });
    
    // Should have 3 * 3 * 3 = 27 combinations
    const expectedCombos = 3 * 3 * 3;
    assert.strictEqual(batchResult.rows, expectedCombos);
    assert.strictEqual(batchResult.cols, 100);
    assert.strictEqual(batchResult.values.length, expectedCombos * 100);
});
