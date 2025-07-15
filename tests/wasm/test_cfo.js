/**
 * WASM binding tests for CFO indicator.
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

test('CFO partial params', () => {
    // Test with default parameters - mirrors check_cfo_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('CFO accuracy', async () => {
    // Test CFO matches expected values from Rust tests - mirrors check_cfo_accuracy
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.cfo;
    
    const result = wasm.cfo_js(
        close,
        expected.defaultParams.period,
        expected.defaultParams.scalar
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values match expected
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-6,
        "CFO last 5 values mismatch"
    );
    
    // Compare full output with Rust
    await compareWithRust('cfo', result, 'close', expected.defaultParams);
});

test('CFO default candles', () => {
    // Test CFO with default parameters - mirrors check_cfo_default_candles
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
});

test('CFO zero period', () => {
    // Test CFO fails with zero period - mirrors check_cfo_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cfo_js(inputData, 0, 100.0);
    }, /Invalid period/);
});

test('CFO period exceeds length', () => {
    // Test CFO fails when period exceeds data length - mirrors check_cfo_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.cfo_js(dataSmall, 10, 100.0);
    }, /Invalid period/);
});

test('CFO very small dataset', () => {
    // Test CFO fails with insufficient data - mirrors check_cfo_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.cfo_js(singlePoint, 14, 100.0);
    }, /Invalid period|Not enough valid data/);
});

test('CFO empty input', () => {
    // Test CFO fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.cfo_js(empty, 14, 100.0);
    }, /No data provided/);
});

test('CFO reinput', () => {
    // Test CFO applied twice (re-input) - mirrors check_cfo_reinput
    const close = new Float64Array(testData.close);
    
    // First pass
    const firstResult = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(firstResult.length, close.length);
    
    // Second pass - apply CFO to CFO output
    const secondResult = wasm.cfo_js(firstResult, 14, 100.0);
    assert.strictEqual(secondResult.length, firstResult.length);
    
    // After warmup period (240), no NaN values should exist
    if (secondResult.length > 240) {
        for (let i = 240; i < secondResult.length; i++) {
            assert(!isNaN(secondResult[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('CFO NaN handling', () => {
    // Test CFO handles NaN values correctly - mirrors check_cfo_nan_handling
    const close = new Float64Array(testData.close);
    
    const result = wasm.cfo_js(close, 14, 100.0);
    assert.strictEqual(result.length, close.length);
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 13), "Expected NaN in warmup period");
});

test('CFO all NaN input', () => {
    // Test CFO with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.cfo_js(allNaN, 14, 100.0);
    }, /All values are NaN/);
});

test('CFO batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set: period=14, scalar=100.0
    const batchResult = wasm.cfo_batch_js(
        close,
        14, 14, 0,      // period range
        100.0, 100.0, 0 // scalar range
    );
    
    // Should match single calculation
    const singleResult = wasm.cfo_js(close, 14, 100.0);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('CFO batch multiple periods', () => {
    // Test batch with multiple period values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple periods: 10, 12, 14
    const batchResult = wasm.cfo_batch_js(
        close,
        10, 14, 2,      // period range
        100.0, 100.0, 0 // scalar range  
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 12, 14];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.cfo_js(close, periods[i], 100.0);
        assertArrayClose(
            rowData, 
            singleResult, 
            1e-10, 
            `Period ${periods[i]} mismatch`
        );
    }
});

test('CFO batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.cfo_batch_metadata_js(
        10, 14, 2,      // period: 10, 12, 14
        50.0, 150.0, 50.0 // scalar: 50.0, 100.0, 150.0
    );
    
    // Should have 3 * 3 = 9 combinations
    // Each combo has 2 values: [period, scalar]
    assert.strictEqual(metadata.length, 9 * 2);
    
    // Check first combination
    assert.strictEqual(metadata[0], 10);    // period
    assert.strictEqual(metadata[1], 50.0);  // scalar
    
    // Check last combination
    assert.strictEqual(metadata[16], 14);   // period
    assert.strictEqual(metadata[17], 150.0); // scalar
});

test('CFO batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.cfo_batch_js(
        close,
        10, 12, 2,      // 2 periods
        50.0, 100.0, 50.0  // 2 scalars
    );
    
    const metadata = wasm.cfo_batch_metadata_js(
        10, 12, 2,
        50.0, 100.0, 50.0
    );
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(batchResult.length, 4 * 50);
    assert.strictEqual(metadata.length, 4 * 2);
});

test('CFO batch unified API', () => {
    // Test ergonomic unified batch API
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        period_range: [10, 14, 2],
        scalar_range: [50.0, 150.0, 50.0]
    };
    
    const result = wasm.cfo_batch(close, config);
    
    // Check structure
    assert(result.values, "Missing values field");
    assert(result.combos, "Missing combos field");
    assert(result.rows, "Missing rows field");
    assert(result.cols, "Missing cols field");
    
    // Should have 3 periods * 3 scalars = 9 combinations
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 9 * 100);
    assert.strictEqual(result.combos.length, 9);
    
    // Check combos structure
    result.combos.forEach(combo => {
        assert(combo.hasOwnProperty('period'), "Combo missing period");
        assert(combo.hasOwnProperty('scalar'), "Combo missing scalar");
    });
});

test('CFO constant values', () => {
    // Test CFO with constant input values
    const constant = new Float64Array(50);
    constant.fill(42.0);
    
    const result = wasm.cfo_js(constant, 14, 100.0);
    assert.strictEqual(result.length, constant.length);
    
    // For constant values, CFO should be 0 after warmup
    // (since forecast = actual for constant series)
    let foundNonNaN = false;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            assert(Math.abs(result[i]) < 1e-10, `CFO should be ~0 for constant series at index ${i}, got ${result[i]}`);
        }
    }
    assert(foundNonNaN, "Should have at least one non-NaN value");
});

test('CFO linear trend', () => {
    // Test CFO with perfect linear trend
    const x = new Float64Array(100);
    for (let i = 0; i < 100; i++) {
        x[i] = i;
    }
    
    const result = wasm.cfo_js(x, 14, 100.0);
    assert.strictEqual(result.length, x.length);
    
    // For perfect linear trend, forecast should equal actual
    // so CFO should be close to 0
    let foundNonNaN = false;
    for (let i = 0; i < result.length; i++) {
        if (!isNaN(result[i])) {
            foundNonNaN = true;
            assert(Math.abs(result[i]) < 1e-10, `CFO should be ~0 for linear trend at index ${i}, got ${result[i]}`);
        }
    }
    assert(foundNonNaN, "Should have at least one non-NaN value");
});