/**
 * WASM binding tests for Alligator indicator.
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
    assertNoNaN 
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

test('Alligator partial params', () => {
    // Test with partial parameters - mirrors check_alligator_partial_params
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Test with jaw_period=14 and lips_offset=2 (overriding defaults)
    const result = wasm.alligator_js(hl2Array, 14, 8, 8, 5, 5, 2);
    
    // Result should be flattened: [jaw, teeth, lips]
    assert.strictEqual(result.length, hl2Array.length * 3);
});

test('Alligator accuracy', async () => {
    // Test Alligator matches expected values from Rust tests - mirrors check_alligator_accuracy
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Expected values from Rust tests
    const expectedLastFiveJaw = [60742.4, 60632.6, 60555.1, 60442.7, 60308.7];
    const expectedLastFiveTeeth = [59908.0, 59757.2, 59684.3, 59653.5, 59621.1];
    const expectedLastFiveLips = [59355.2, 59371.7, 59376.2, 59334.1, 59316.2];
    
    const result = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3); // Default params
    
    // Result is flattened: [jaw array, teeth array, lips array]
    const len = hl2Array.length;
    const jaw = result.slice(0, len);
    const teeth = result.slice(len, 2 * len);
    const lips = result.slice(2 * len, 3 * len);
    
    assert.strictEqual(jaw.length, len);
    assert.strictEqual(teeth.length, len);
    assert.strictEqual(lips.length, len);
    
    // Check last 5 values match expected
    const jawLast5 = jaw.slice(-5);
    const teethLast5 = teeth.slice(-5);
    const lipsLast5 = lips.slice(-5);
    
    assertArrayClose(
        jawLast5,
        expectedLastFiveJaw,
        0.1, // Using 1e-1 tolerance as in Rust tests
        "Alligator jaw last 5 values mismatch"
    );
    
    assertArrayClose(
        teethLast5,
        expectedLastFiveTeeth,
        0.1,
        "Alligator teeth last 5 values mismatch"
    );
    
    assertArrayClose(
        lipsLast5,
        expectedLastFiveLips,
        0.1,
        "Alligator lips last 5 values mismatch"
    );
});

test('Alligator default candles', () => {
    // Test Alligator with default parameters - mirrors check_alligator_default_candles
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    const result = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    assert.strictEqual(result.length, hl2Array.length * 3);
});

test('Alligator zero jaw period', () => {
    // Test Alligator fails with zero jaw period - mirrors check_alligator_zero_jaw_period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alligator_js(data, 0, 8, 8, 5, 5, 3);
    }, /Invalid jaw period/);
});

test('Alligator zero teeth period', () => {
    // Test Alligator fails with zero teeth period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alligator_js(data, 13, 8, 0, 5, 5, 3);
    }, /Invalid teeth period/);
});

test('Alligator zero lips period', () => {
    // Test Alligator fails with zero lips period
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alligator_js(data, 13, 8, 8, 5, 0, 3);
    }, /Invalid lips period/);
});

test('Alligator period exceeds length', () => {
    // Test Alligator fails when period exceeds data length
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.alligator_js(dataSmall, 10, 8, 8, 5, 5, 3);
    }, /Invalid jaw period/);
});

test('Alligator offset exceeds length', () => {
    // Test Alligator fails when offset exceeds data length
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.alligator_js(data, 3, 10, 3, 5, 3, 3);
    }, /Invalid jaw offset/);
    
    assert.throws(() => {
        wasm.alligator_js(data, 3, 3, 3, 10, 3, 3);
    }, /Invalid teeth offset/);
    
    assert.throws(() => {
        wasm.alligator_js(data, 3, 3, 3, 3, 3, 10);
    }, /Invalid lips offset/);
});

test('Alligator all NaN input', () => {
    // Test Alligator with all NaN values
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.alligator_js(allNaN, 13, 8, 8, 5, 5, 3);
    }, /All values are NaN/);
});

test('Alligator reinput', () => {
    // Test Alligator applied to jaw output - mirrors check_alligator_with_slice_data_reinput
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // First pass
    const firstResult = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const len = hl2Array.length;
    const firstJaw = firstResult.slice(0, len);
    
    // Second pass - apply Alligator to jaw output
    const secondResult = wasm.alligator_js(new Float64Array(firstJaw), 13, 8, 8, 5, 5, 3);
    assert.strictEqual(secondResult.length, firstJaw.length * 3);
});

test('Alligator NaN handling', () => {
    // Test Alligator handles NaN values correctly - mirrors check_alligator_nan_handling
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    const result = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const len = hl2Array.length;
    const jaw = result.slice(0, len);
    const teeth = result.slice(len, 2 * len);
    const lips = result.slice(2 * len, 3 * len);
    
    // After warmup period (50), no NaN values should exist
    if (jaw.length > 50) {
        for (let i = 50; i < jaw.length; i++) {
            assert(!isNaN(jaw[i]), `Found unexpected NaN in jaw at index ${i}`);
            assert(!isNaN(teeth[i]), `Found unexpected NaN in teeth at index ${i}`);
            assert(!isNaN(lips[i]), `Found unexpected NaN in lips at index ${i}`);
        }
    }
});

test('Alligator batch single parameter set', () => {
    // Test batch with single parameter combination
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Single parameter set with defaults
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 13, 0,  // jaw_period range
        8, 8, 0,    // jaw_offset range
        8, 8, 0,    // teeth_period range
        5, 5, 0,    // teeth_offset range
        5, 5, 0,    // lips_period range
        3, 3, 0     // lips_offset range
    );
    
    // Should match single calculation
    const singleResult = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    
    // Batch result is flattened: [all jaw values, all teeth values, all lips values]
    // For single parameter set, it should be same as single result
    assert.strictEqual(batchResult.length, singleResult.length);
    
    // Compare each section
    const len = hl2Array.length;
    const batchJaw = batchResult.slice(0, len);
    const batchTeeth = batchResult.slice(len, 2 * len);
    const batchLips = batchResult.slice(2 * len, 3 * len);
    
    const singleJaw = singleResult.slice(0, len);
    const singleTeeth = singleResult.slice(len, 2 * len);
    const singleLips = singleResult.slice(2 * len, 3 * len);
    
    assertArrayClose(batchJaw, singleJaw, 1e-10, "Batch vs single jaw mismatch");
    assertArrayClose(batchTeeth, singleTeeth, 1e-10, "Batch vs single teeth mismatch");
    assertArrayClose(batchLips, singleLips, 1e-10, "Batch vs single lips mismatch");
});

test('Alligator batch multiple jaw periods', () => {
    // Test batch with multiple jaw period values
    const hl2 = testData.high.slice(0, 100).map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Multiple jaw periods: 13, 15, 17
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 17, 2,  // jaw_period range
        8, 8, 0,    // jaw_offset range
        8, 8, 0,    // teeth_period range
        5, 5, 0,    // teeth_offset range
        5, 5, 0,    // lips_period range
        3, 3, 0     // lips_offset range
    );
    
    // Should have 3 rows * 100 cols * 3 arrays = 900 values
    assert.strictEqual(batchResult.length, 3 * 100 * 3);
    
    // Verify each row matches individual calculation
    const jawPeriods = [13, 15, 17];
    const totalArraySize = 3 * 100; // 3 combos * 100 data points
    
    for (let i = 0; i < jawPeriods.length; i++) {
        const rowStartJaw = i * 100;
        const rowEndJaw = rowStartJaw + 100;
        const rowJaw = batchResult.slice(rowStartJaw, rowEndJaw);
        
        const rowStartTeeth = totalArraySize + i * 100;
        const rowEndTeeth = rowStartTeeth + 100;
        const rowTeeth = batchResult.slice(rowStartTeeth, rowEndTeeth);
        
        const rowStartLips = 2 * totalArraySize + i * 100;
        const rowEndLips = rowStartLips + 100;
        const rowLips = batchResult.slice(rowStartLips, rowEndLips);
        
        const singleResult = wasm.alligator_js(hl2Array, jawPeriods[i], 8, 8, 5, 5, 3);
        const singleJaw = singleResult.slice(0, 100);
        const singleTeeth = singleResult.slice(100, 200);
        const singleLips = singleResult.slice(200, 300);
        
        assertArrayClose(
            rowJaw, 
            singleJaw, 
            1e-10, 
            `Jaw period ${jawPeriods[i]} mismatch`
        );
        assertArrayClose(
            rowTeeth, 
            singleTeeth, 
            1e-10, 
            `Teeth period ${jawPeriods[i]} mismatch`
        );
        assertArrayClose(
            rowLips, 
            singleLips, 
            1e-10, 
            `Lips period ${jawPeriods[i]} mismatch`
        );
    }
});

test('Alligator batch metadata', () => {
    // Test metadata function returns correct parameter combinations
    const metadata = wasm.alligator_batch_metadata_js(
        13, 14, 1,  // jaw_period: 13, 14
        8, 9, 1,    // jaw_offset: 8, 9
        8, 8, 0,    // teeth_period: 8
        5, 5, 0,    // teeth_offset: 5
        5, 5, 0,    // lips_period: 5
        3, 3, 0     // lips_offset: 3
    );
    
    // Should have 2 * 2 * 1 * 1 * 1 * 1 = 4 combinations
    // Each combo has 6 values: [jaw_period, jaw_offset, teeth_period, teeth_offset, lips_period, lips_offset]
    assert.strictEqual(metadata.length, 4 * 6);
    
    // Check first combination
    assert.strictEqual(metadata[0], 13);  // jaw_period
    assert.strictEqual(metadata[1], 8);   // jaw_offset
    assert.strictEqual(metadata[2], 8);   // teeth_period
    assert.strictEqual(metadata[3], 5);   // teeth_offset
    assert.strictEqual(metadata[4], 5);   // lips_period
    assert.strictEqual(metadata[5], 3);   // lips_offset
    
    // Check last combination
    assert.strictEqual(metadata[18], 14); // jaw_period
    assert.strictEqual(metadata[19], 9);  // jaw_offset
    assert.strictEqual(metadata[20], 8);  // teeth_period
    assert.strictEqual(metadata[21], 5);  // teeth_offset
    assert.strictEqual(metadata[22], 5);  // lips_period
    assert.strictEqual(metadata[23], 3);  // lips_offset
});

test('Alligator batch full parameter sweep', () => {
    // Test full parameter sweep matching expected structure
    const hl2 = testData.high.slice(0, 50).map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    const batchResult = wasm.alligator_batch_js(
        hl2Array,
        13, 14, 1,  // 2 jaw periods
        8, 8, 0,    // 1 jaw offset
        8, 8, 0,    // 1 teeth period
        5, 5, 0,    // 1 teeth offset
        5, 6, 1,    // 2 lips periods
        3, 3, 0     // 1 lips offset
    );
    
    const metadata = wasm.alligator_batch_metadata_js(
        13, 14, 1,
        8, 8, 0,
        8, 8, 0,
        5, 5, 0,
        5, 6, 1,
        3, 3, 0
    );
    
    // Should have 2 * 1 * 1 * 1 * 2 * 1 = 4 combinations
    const numCombos = metadata.length / 6;
    assert.strictEqual(numCombos, 4);
    assert.strictEqual(batchResult.length, 4 * 50 * 3); // 4 combos * 50 data points * 3 arrays
});

test('Alligator batch edge cases', () => {
    // Test edge cases for batch processing
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.alligator_batch_js(
        data,
        5, 5, 1,
        3, 3, 1,
        4, 4, 1,
        2, 2, 1,
        3, 3, 1,
        1, 1, 1
    );
    
    assert.strictEqual(singleBatch.length, 15 * 3); // 1 combo * 15 data points * 3 arrays
    
    // Empty data should throw
    assert.throws(() => {
        wasm.alligator_batch_js(
            new Float64Array([]),
            13, 13, 0,
            8, 8, 0,
            8, 8, 0,
            5, 5, 0,
            5, 5, 0,
            3, 3, 0
        );
    }, /All values are NaN/);
});

// New API tests
test('Alligator batch - new ergonomic API with single parameter', () => {
    // Test the new API with a single parameter combination
    const hl2 = testData.high.map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    const result = wasm.alligator_batch(hl2Array, {
        jaw_period_range: [13, 13, 0],
        jaw_offset_range: [8, 8, 0],
        teeth_period_range: [8, 8, 0],
        teeth_offset_range: [5, 5, 0],
        lips_period_range: [5, 5, 0],
        lips_offset_range: [3, 3, 0]
    });
    
    // Verify structure
    assert(result.jaw, 'Should have jaw array');
    assert(result.teeth, 'Should have teeth array');
    assert(result.lips, 'Should have lips array');
    assert(result.combos, 'Should have combos array');
    assert(typeof result.rows === 'number', 'Should have rows count');
    assert(typeof result.cols === 'number', 'Should have cols count');
    
    // Verify dimensions
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, hl2Array.length);
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.jaw.length, hl2Array.length);
    assert.strictEqual(result.teeth.length, hl2Array.length);
    assert.strictEqual(result.lips.length, hl2Array.length);
    
    // Verify parameters
    const combo = result.combos[0];
    assert.strictEqual(combo.jaw_period, 13);
    assert.strictEqual(combo.jaw_offset, 8);
    assert.strictEqual(combo.teeth_period, 8);
    assert.strictEqual(combo.teeth_offset, 5);
    assert.strictEqual(combo.lips_period, 5);
    assert.strictEqual(combo.lips_offset, 3);
    
    // Compare with old API
    const oldResult = wasm.alligator_js(hl2Array, 13, 8, 8, 5, 5, 3);
    const oldJaw = oldResult.slice(0, hl2Array.length);
    const oldTeeth = oldResult.slice(hl2Array.length, 2 * hl2Array.length);
    const oldLips = oldResult.slice(2 * hl2Array.length, 3 * hl2Array.length);
    
    for (let i = 0; i < hl2Array.length; i++) {
        if (isNaN(oldJaw[i]) && isNaN(result.jaw[i])) {
            continue; // Both NaN is OK
        }
        assert(Math.abs(oldJaw[i] - result.jaw[i]) < 1e-10,
               `Jaw value mismatch at index ${i}`);
        assert(Math.abs(oldTeeth[i] - result.teeth[i]) < 1e-10,
               `Teeth value mismatch at index ${i}`);
        assert(Math.abs(oldLips[i] - result.lips[i]) < 1e-10,
               `Lips value mismatch at index ${i}`);
    }
});

test('Alligator batch - new API with multiple parameters', () => {
    // Test with multiple parameter combinations
    const hl2 = testData.high.slice(0, 50).map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    const result = wasm.alligator_batch(hl2Array, {
        jaw_period_range: [13, 14, 1],     // 13, 14
        jaw_offset_range: [8, 8, 0],       // 8
        teeth_period_range: [8, 8, 0],     // 8
        teeth_offset_range: [5, 5, 0],     // 5
        lips_period_range: [5, 6, 1],      // 5, 6
        lips_offset_range: [3, 3, 0]       // 3
    });
    
    // Should have 2 * 1 * 1 * 1 * 2 * 1 = 4 combinations
    assert.strictEqual(result.rows, 4);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.combos.length, 4);
    assert.strictEqual(result.jaw.length, 200);  // 4 rows * 50 cols
    assert.strictEqual(result.teeth.length, 200);
    assert.strictEqual(result.lips.length, 200);
    
    // Verify each combo
    const expectedCombos = [
        { jaw_period: 13, jaw_offset: 8, teeth_period: 8, teeth_offset: 5, lips_period: 5, lips_offset: 3 },
        { jaw_period: 13, jaw_offset: 8, teeth_period: 8, teeth_offset: 5, lips_period: 6, lips_offset: 3 },
        { jaw_period: 14, jaw_offset: 8, teeth_period: 8, teeth_offset: 5, lips_period: 5, lips_offset: 3 },
        { jaw_period: 14, jaw_offset: 8, teeth_period: 8, teeth_offset: 5, lips_period: 6, lips_offset: 3 }
    ];
    
    for (let i = 0; i < expectedCombos.length; i++) {
        assert.strictEqual(result.combos[i].jaw_period, expectedCombos[i].jaw_period);
        assert.strictEqual(result.combos[i].jaw_offset, expectedCombos[i].jaw_offset);
        assert.strictEqual(result.combos[i].teeth_period, expectedCombos[i].teeth_period);
        assert.strictEqual(result.combos[i].teeth_offset, expectedCombos[i].teeth_offset);
        assert.strictEqual(result.combos[i].lips_period, expectedCombos[i].lips_period);
        assert.strictEqual(result.combos[i].lips_offset, expectedCombos[i].lips_offset);
    }
});

test('Alligator batch - new API error handling', () => {
    const hl2 = testData.high.slice(0, 10).map((h, i) => (h + testData.low[i]) / 2);
    const hl2Array = new Float64Array(hl2);
    
    // Invalid config structure
    assert.throws(() => {
        wasm.alligator_batch(hl2Array, {
            jaw_period_range: [13, 13], // Missing step
            jaw_offset_range: [8, 8, 0],
            teeth_period_range: [8, 8, 0],
            teeth_offset_range: [5, 5, 0],
            lips_period_range: [5, 5, 0],
            lips_offset_range: [3, 3, 0]
        });
    }, /Invalid config/);
    
    // Missing required field
    assert.throws(() => {
        wasm.alligator_batch(hl2Array, {
            jaw_period_range: [13, 13, 0],
            jaw_offset_range: [8, 8, 0],
            teeth_period_range: [8, 8, 0],
            teeth_offset_range: [5, 5, 0],
            lips_period_range: [5, 5, 0]
            // Missing lips_offset_range
        });
    }, /Invalid config/);
    
    // Invalid data type
    assert.throws(() => {
        wasm.alligator_batch(hl2Array, {
            jaw_period_range: "invalid",
            jaw_offset_range: [8, 8, 0],
            teeth_period_range: [8, 8, 0],
            teeth_offset_range: [5, 5, 0],
            lips_period_range: [5, 5, 0],
            lips_offset_range: [3, 3, 0]
        });
    }, /Invalid config/);
});

test.after(() => {
    console.log('Alligator WASM tests completed');
});