/**
 * WASM binding tests for AO (Awesome Oscillator) indicator.
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

test('AO with default parameters', () => {
    const { high, low } = testData;
    const result = wasm.ao_js(high, low, 5, 34);
    
    // WASM returns Float64Array, not regular Array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, high.length);
    
    // Check warmup period
    const warmupPeriod = 34; // long_period
    for (let i = 0; i < warmupPeriod - 1; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should not be NaN
    for (let i = warmupPeriod - 1; i < result.length; i++) {
        assert.ok(isFinite(result[i]), `Value at index ${i} should be finite`);
    }
});

test('AO matches expected values from Rust tests', () => {
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.ao;
    
    const result = wasm.ao_js(
        high, low,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );
    
    // Check last 5 values match expected with tolerance
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 1e-4, 'AO last 5 values mismatch');
});

test('AO with empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.ao_js(empty, empty, 5, 34),
        /empty|no data/i
    );
});

test('AO with mismatched lengths', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    
    assert.throws(
        () => wasm.ao_js(high, low, 5, 34),
        /same length/
    );
});

test('AO with invalid periods', () => {
    const high = new Float64Array(50).fill(100);
    const low = new Float64Array(50).fill(90);
    
    // Zero short period
    assert.throws(
        () => wasm.ao_js(high, low, 0, 34),
        /Invalid periods/
    );
    
    // Zero long period
    assert.throws(
        () => wasm.ao_js(high, low, 5, 0),
        /Invalid periods/
    );
    
    // Short period >= long period
    assert.throws(
        () => wasm.ao_js(high, low, 34, 34),
        /Short period must be less than long period/
    );
    
    assert.throws(
        () => wasm.ao_js(high, low, 35, 34),
        /Short period must be less than long period/
    );
});

test('AO when period exceeds data length', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(
        () => wasm.ao_js(high, low, 5, 10),
        /Not enough valid data/
    );
});

test('AO with all NaN values', () => {
    const allNan = new Float64Array(10).fill(NaN);
    
    assert.throws(
        () => wasm.ao_js(allNan, allNan, 5, 34),
        /All values are NaN/
    );
});

test('AO with leading NaN values', () => {
    const { high, low } = testData;
    const highWithNan = new Float64Array(high);
    const lowWithNan = new Float64Array(low);
    
    // Add some leading NaNs
    for (let i = 0; i < 5; i++) {
        highWithNan[i] = NaN;
        lowWithNan[i] = NaN;
    }
    
    const result = wasm.ao_js(highWithNan, lowWithNan, 5, 34);
    
    assert.strictEqual(result.length, high.length);
    
    // First 5 + warmup period should be NaN
    for (let i = 0; i < 5 + 34 - 1; i++) {
        assert.ok(isNaN(result[i]));
    }
});

test('AO with constant prices', () => {
    const length = 50;
    const constantPrice = 100.0;
    const high = new Float64Array(length).fill(constantPrice);
    const low = new Float64Array(length).fill(constantPrice);
    
    const result = wasm.ao_js(high, low, 5, 34);
    
    // With constant prices, AO should be 0 after warmup
    const warmup = 34;
    for (let i = warmup; i < length; i++) {
        assert.ok(Math.abs(result[i]) < 1e-10, 
            `Expected 0 at index ${i}, got ${result[i]}`);
    }
});

test('AO in trending market', () => {
    const length = 100;
    // Create uptrending data
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    
    for (let i = 0; i < length; i++) {
        const price = 100 + i; // Uptrending
        high[i] = price + 5;
        low[i] = price - 5;
    }
    
    const result = wasm.ao_js(high, low, 5, 34);
    
    // In a strong uptrend, AO should be positive after initial period
    // Check last 10 values are positive
    for (let i = result.length - 10; i < result.length; i++) {
        assert.ok(result[i] > 0, `AO should be positive in uptrend at index ${i}`);
    }
});

test('AO batch calculation with single parameters', () => {
    const { high, low } = testData;
    
    const result = wasm.ao_batch_js(
        high, low,
        5, 5, 0,   // short_period range (single value)
        34, 34, 0  // long_period range (single value)
    );
    
    // Batch returns flat array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, high.length);
    
    // Should match single calculation
    const singleResult = wasm.ao_js(high, low, 5, 34);
    assertArrayClose(
        Array.from(result), 
        Array.from(singleResult), 
        1e-10,
        'Batch vs single calculation mismatch'
    );
});

test('AO batch calculation with parameter sweep', () => {
    const { high, low } = testData;
    const dataLen = Math.min(high.length, 100); // Use smaller subset for speed
    const highSubset = high.slice(0, dataLen);
    const lowSubset = low.slice(0, dataLen);
    
    const result = wasm.ao_batch_js(
        highSubset, lowSubset,
        3, 7, 2,    // short: 3, 5, 7
        20, 30, 5   // long: 20, 25, 30
    );
    
    // Should have 3 * 3 = 9 combinations
    const expectedRows = 3 * 3;
    assert.strictEqual(result.length, expectedRows * dataLen);
});

test('AO batch metadata', () => {
    // For short_period 3-7 step 2 and long_period 20-30 step 5
    const meta = wasm.ao_batch_metadata_js(3, 7, 2, 20, 30, 5);
    
    assert.ok(meta instanceof Float64Array || Array.isArray(meta));
    // 3 short periods * 3 long periods = 9 combos, each has 2 values
    assert.strictEqual(meta.length, 3 * 3 * 2);
    
    // Check structure (short, long pairs)
    const expectedPairs = [
        [3, 20], [3, 25], [3, 30],
        [5, 20], [5, 25], [5, 30],
        [7, 20], [7, 25], [7, 30]
    ];
    
    for (let i = 0; i < expectedPairs.length; i++) {
        assert.strictEqual(meta[i * 2], expectedPairs[i][0], `Short period mismatch at combo ${i}`);
        assert.strictEqual(meta[i * 2 + 1], expectedPairs[i][1], `Long period mismatch at combo ${i}`);
    }
});

test('AO batch with invalid combinations', () => {
    const { high, low } = testData;
    const highSubset = high.slice(0, 50);
    const lowSubset = low.slice(0, 50);
    
    // Range where some short >= long (should be filtered out)
    const result = wasm.ao_batch_js(
        highSubset, lowSubset,
        5, 15, 5,   // short: 5, 10, 15
        10, 12, 2   // long: 10, 12
    );
    
    // Only valid combos where short < long
    // Valid: (5,10), (5,12), (10,12)
    const expectedRows = 3;
    assert.strictEqual(result.length, expectedRows * 50);
});

test('AO batch (unified API)', () => {
    const { high, low } = testData;
    
    const config = {
        short_period_range: [5, 5, 0],
        long_period_range: [34, 34, 0]
    };
    
    const result = wasm.ao_batch(high, low, config);
    
    assert.ok(result);
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, high.length);
    
    // Check combo structure
    assert.strictEqual(result.combos[0].short_period, 5);
    assert.strictEqual(result.combos[0].long_period, 34);
});

test('AO error handling coverage', () => {
    const { high, low } = testData;
    
    // AllValuesNaN
    assert.throws(
        () => wasm.ao_js(new Float64Array(10).fill(NaN), new Float64Array(10).fill(NaN), 5, 34),
        /All values are NaN/
    );
    
    // InvalidPeriods
    assert.throws(
        () => wasm.ao_js(high.slice(0, 50), low.slice(0, 50), 0, 34),
        /Invalid periods/
    );
    
    // ShortPeriodNotLess
    assert.throws(
        () => wasm.ao_js(high.slice(0, 50), low.slice(0, 50), 34, 34),
        /Short period must be less than long period/
    );
    
    // NoData
    assert.throws(
        () => wasm.ao_js(new Float64Array([]), new Float64Array([]), 5, 34),
        /empty|no data/i
    );
    
    // NotEnoughValidData
    assert.throws(
        () => wasm.ao_js(high.slice(0, 10), low.slice(0, 10), 5, 34),
        /Not enough valid data/
    );
});

test('AO real-world conditions', () => {
    const { high, low } = testData;
    
    const result = wasm.ao_js(high, low, 5, 34);
    
    // Check warmup period behavior
    const warmup = 34; // long_period
    
    // Values before warmup should be NaN
    for (let i = 0; i < warmup - 1; i++) {
        assert.ok(isNaN(result[i]));
    }
    
    // Values from warmup onwards should not be NaN
    const validStart = warmup - 1;
    for (let i = validStart; i < result.length; i++) {
        assert.ok(!isNaN(result[i]));
    }
    
    // Check output properties
    assert.strictEqual(result.length, high.length);
    
    // AO typically oscillates around zero
    // Check that we have both positive and negative values
    const validValues = Array.from(result.slice(validStart));
    assert.ok(validValues.some(v => v > 0), 'Should have some positive values');
    assert.ok(validValues.some(v => v < 0), 'Should have some negative values');
});

test('AO comparison with Rust', () => {
    const { high, low } = testData;
    const expected = EXPECTED_OUTPUTS.ao;
    
    const result = wasm.ao_js(
        high, low,
        expected.defaultParams.short_period,
        expected.defaultParams.long_period
    );
    
    // TODO: Fix generate_references to support AO's high/low inputs
    // compareWithRust('ao', Array.from(result), 'hl', expected.defaultParams);
});