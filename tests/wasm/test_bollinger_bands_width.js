/**
 * WASM binding tests for Bollinger Bands Width indicator.
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

test('BBW partial params', () => {
    // Test with custom parameters - mirrors check_bbw_partial_params
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(
        close,
        22,     // period
        2.2,    // devup
        2.0,    // devdn
        "ema",  // matype
        null    // devtype - use default
    );
    
    assert.strictEqual(result.length, close.length);
});

test('BBW default params', () => {
    // Test with default parameters - mirrors check_bbw_default
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(
        close,
        20,     // period
        2.0,    // devup
        2.0,    // devdn
        null,   // matype - use default "sma"
        null    // devtype - use default 0
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // After warmup should have values
    if (result.length > 240) {
        assertNoNaN(result.slice(240), "Expected no NaN after sufficient warmup");
    }
});

test('BBW accuracy', async () => {
    // Test BBW matches expected values
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.bollinger_bands_width || {};
    
    const result = wasm.bollinger_bands_width_js(
        close,
        expected.default_params?.period || 20,
        expected.default_params?.devup || 2.0,
        expected.default_params?.devdn || 2.0,
        expected.default_params?.matype || "sma",
        expected.default_params?.devtype || 0
    );
    
    assert.strictEqual(result.length, close.length);
    
    // Check last 5 values if available
    if (expected.last_5_values) {
        const last5 = result.slice(-5);
        assertArrayClose(
            last5,
            expected.last_5_values,
            1e-8,
            "BBW last 5 values mismatch"
        );
    }
    
    // Compare full output with Rust
    await compareWithRust('bollinger_bands_width', result, 'close', {
        period: expected.default_params?.period || 20,
        devup: expected.default_params?.devup || 2.0,
        devdn: expected.default_params?.devdn || 2.0,
        matype: expected.default_params?.matype || "sma",
        devtype: expected.default_params?.devtype || 0
    });
});

test('BBW zero period', () => {
    // Test BBW fails with zero period - mirrors check_bbw_zero_period
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(inputData, 0, 2.0, 2.0, null, null);
    }, /Invalid period|period/);
});

test('BBW period exceeds length', () => {
    // Test BBW fails when period exceeds data length - mirrors check_bbw_period_exceeds_length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(dataSmall, 10, 2.0, 2.0, null, null);
    }, /Invalid period|period/);
});

test('BBW very small dataset', () => {
    // Test BBW fails with insufficient data - mirrors check_bbw_very_small_dataset
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(singlePoint, 20, 2.0, 2.0, null, null);
    }, /Invalid period|Not enough valid data/);
});

test('BBW empty input', () => {
    // Test BBW fails with empty input
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(empty, 20, 2.0, 2.0, null, null);
    }, /Empty data/);
});

test('BBW all NaN input', () => {
    // Test BBW with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.bollinger_bands_width_js(allNaN, 20, 2.0, 2.0, null, null);
    }, /All values are NaN/);
});

test('BBW different MA types', () => {
    // Test BBW with different moving average types
    const close = new Float64Array(testData.close.slice(0, 100));
    const matypes = ["sma", "ema", "wma", "dema", "tema"];
    
    for (const matype of matypes) {
        const result = wasm.bollinger_bands_width_js(
            close,
            14,
            2.0,
            2.0,
            matype,
            0
        );
        
        assert.strictEqual(result.length, 100);
        // Check warmup period
        for (let i = 0; i < 13; i++) {
            assert(isNaN(result[i]), `Expected NaN at index ${i} for ${matype}`);
        }
    }
});

test('BBW different deviation types', () => {
    // Test BBW with different deviation types
    const close = new Float64Array(testData.close.slice(0, 100));
    const devtypes = [0, 1, 2]; // stddev, mean_ad, median_ad
    
    for (const devtype of devtypes) {
        const result = wasm.bollinger_bands_width_js(
            close,
            14,
            2.0,
            2.0,
            "sma",
            devtype
        );
        
        assert.strictEqual(result.length, 100);
    }
});

test('BBW NaN handling', () => {
    // Test BBW handles NaN values correctly
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, null, null);
    assert.strictEqual(result.length, close.length);
    
    // First period-1 values should be NaN
    assertAllNaN(result.slice(0, 19), "Expected NaN in warmup period");
    
    // After warmup period (240), no NaN values should exist
    if (result.length > 240) {
        for (let i = 240; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('BBW batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close);
    
    // Single parameter set
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        20, 20, 0,      // period range
        2.0, 2.0, 0,    // devup range
        2.0, 2.0, 0     // devdn range
    );
    
    // Should match single calculation
    const singleResult = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, null, null);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('BBW batch multiple parameters', () => {
    // Test batch with multiple parameter values
    const close = new Float64Array(testData.close.slice(0, 100)); // Use smaller dataset for speed
    
    // Multiple parameters
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        10, 30, 10,     // period: 10, 20, 30
        1.5, 2.5, 0.5,  // devup: 1.5, 2.0, 2.5
        2.0, 2.0, 0     // devdn: 2.0
    );
    
    // Should have 3 * 3 * 1 = 9 rows * 100 cols = 900 values
    assert.strictEqual(batchResult.length, 9 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 20, 30];
    const devups = [1.5, 2.0, 2.5];
    let rowIdx = 0;
    
    for (const period of periods) {
        for (const devup of devups) {
            const rowStart = rowIdx * 100;
            const rowEnd = rowStart + 100;
            const rowData = batchResult.slice(rowStart, rowEnd);
            
            const singleResult = wasm.bollinger_bands_width_js(
                close, period, devup, 2.0, null, null
            );
            
            assertArrayClose(
                rowData, 
                singleResult, 
                1e-10, 
                `Period ${period}, devup ${devup} mismatch`
            );
            
            rowIdx++;
        }
    }
});

test('BBW batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.bollinger_bands_width_batch_metadata_js(
        10, 20, 10,     // period: 10, 20
        1.5, 2.0, 0.5,  // devup: 1.5, 2.0
        2.0, 3.0, 1.0   // devdn: 2.0, 3.0
    );
    
    // Should have 2 * 2 * 2 = 8 combinations
    // Each combo has 3 values: [period, devup, devdn]
    assert.strictEqual(metadata.length, 8 * 3);
    
    // Check first combination
    assert.strictEqual(metadata[0], 10);   // period
    assert.strictEqual(metadata[1], 1.5);  // devup
    assert.strictEqual(metadata[2], 2.0);  // devdn
    
    // Check last combination
    assert.strictEqual(metadata[21], 20);  // period
    assert.strictEqual(metadata[22], 2.0); // devup
    assert.strictEqual(metadata[23], 3.0); // devdn
});

test('BBW batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const batchResult = wasm.bollinger_bands_width_batch_js(
        close,
        10, 15, 5,      // 2 periods
        2.0, 2.5, 0.5,  // 2 devups
        1.5, 1.5, 0     // 1 devdn
    );
    
    const metadata = wasm.bollinger_bands_width_batch_metadata_js(
        10, 15, 5,
        2.0, 2.5, 0.5,
        1.5, 1.5, 0
    );
    
    // Should have 2 * 2 * 1 = 4 combinations
    const numCombos = metadata.length / 3;
    assert.strictEqual(numCombos, 4);
    assert.strictEqual(batchResult.length, 4 * 50);
    
    // Verify structure
    for (let combo = 0; combo < numCombos; combo++) {
        const period = metadata[combo * 3];
        const devup = metadata[combo * 3 + 1];
        const devdn = metadata[combo * 3 + 2];
        
        const rowStart = combo * 50;
        const rowData = batchResult.slice(rowStart, rowStart + 50);
        
        // First period-1 values should be NaN
        for (let i = 0; i < period - 1; i++) {
            assert(isNaN(rowData[i]), `Expected NaN at warmup index ${i} for period ${period}`);
        }
        
        // After warmup should have values
        for (let i = period - 1; i < 50; i++) {
            assert(!isNaN(rowData[i]), `Unexpected NaN at index ${i} for period ${period}`);
        }
    }
});

test('BBW batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Single value sweep
    const singleBatch = wasm.bollinger_bands_width_batch_js(
        close,
        5, 5, 1,
        2.0, 2.0, 0.1,
        2.0, 2.0, 1.0
    );
    
    assert.strictEqual(singleBatch.length, 10);
    
    // Step larger than range
    const largeBatch = wasm.bollinger_bands_width_batch_js(
        close,
        5, 7, 10, // Step larger than range
        2.0, 2.0, 0,
        2.0, 2.0, 0
    );
    
    // Should only have period=5
    assert.strictEqual(largeBatch.length, 10);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.bollinger_bands_width_batch_js(
            new Float64Array([]),
            10, 10, 0,
            2.0, 2.0, 0,
            2.0, 2.0, 0
        );
    }, /Empty data/);
});

// New API tests
test('BBW batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const close = new Float64Array(testData.close);
    
    const result = wasm.bollinger_bands_width_batch(close, {
        period_range: [20, 20, 0],
        devup_range: [2.0, 2.0, 0],
        devdn_range: [2.0, 2.0, 0],
        matype: "sma",
        devtype: 0
    });
    
    // Verify structure
    assert(result.values, 'Should have values array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.values.length, close.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.period, 20);
    assert.strictEqual(combo.devup, 2.0);
    assert.strictEqual(combo.devdn, 2.0);
    assert.strictEqual(combo.matype, "sma");
    assert.strictEqual(combo.devtype, 0);
    
    // Compare with old API
    const oldResult = wasm.bollinger_bands_width_js(close, 20, 2.0, 2.0, "sma", 0);
    for (let i = 0; i < oldResult.length; i++) {
        if (isNaN(oldResult[i]) && isNaN(result.values[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldResult[i] - result.values[i]) < 1e-10,
               `Value mismatch at index ${i}`);
    }
});

test('BBW batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const result = wasm.bollinger_bands_width_batch(close, {
        period_range: [10, 20, 10],     // 10, 20
        devup_range: [1.5, 2.0, 0.5],   // 1.5, 2.0
        devdn_range: [2.0, 2.0, 0],     // 2.0
        matype: "ema",
        devtype: 1
    });
    
    // Should have 2 * 2 * 1 = 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.values.length, 200);
    
    // Verify each combo
    const expectedCombos = [
        { period: 10, devup: 1.5, devdn: 2.0 },
        { period: 10, devup: 2.0, devdn: 2.0 },
        { period: 20, devup: 1.5, devdn: 2.0 },
        { period: 20, devup: 2.0, devdn: 2.0 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].period, expectedCombos[i].period);
        assert.strictEqual(result.combos[i].devup, expectedCombos[i].devup);
        assert.strictEqual(result.combos[i].devdn, expectedCombos[i].devdn);
        assert.strictEqual(result.combos[i].matype, "ema");
        assert.strictEqual(result.combos[i].devtype, 1);
    }
});

test('BBW batch - new API error handling', () => {
    const close = new Float64Array(testData.close.slice(0, 10));
    
    // Invalid config structure
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: [10, 10], // Missing step
            devup_range: [2.0, 2.0, 0],
            devdn_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: [10, 10, 0],
            devup_range: [2.0, 2.0, 0]
            // Missing devdn_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.bollinger_bands_width_batch(close, {
            period_range: "invalid",
            devup_range: [2.0, 2.0, 0],
            devdn_range: [2.0, 2.0, 0]
        });
    }, /Invalid config/);
});

test.after(() => {
    console.log('Bollinger Bands Width WASM tests completed');
});