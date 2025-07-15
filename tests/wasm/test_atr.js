/**
 * WASM binding tests for ATR (Average True Range) indicator.
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

test('ATR with default parameters', () => {
    const { high, low, close } = testData;
    const result = wasm.atr(high, low, close, 14);
    
    // WASM returns Float64Array, not regular Array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    // Check warmup period
    const warmupPeriod = 14 - 1; // length - 1
    for (let i = 0; i < warmupPeriod; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    // Values after warmup should not be NaN
    for (let i = warmupPeriod; i < result.length; i++) {
        assert.ok(isFinite(result[i]), `Value at index ${i} should be finite`);
        assert.ok(result[i] >= 0, `ATR at index ${i} should be non-negative`);
    }
});

test('ATR matches expected values from Rust tests', () => {
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.atr;
    
    const result = wasm.atr(
        high, low, close,
        expected.defaultParams.length
    );
    
    // Check last 5 values match expected with tolerance
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 1e-2, 'ATR last 5 values mismatch');
});

test('ATR with empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.atr(empty, empty, empty, 14),
        /No candles|no data/
    );
});

test('ATR with mismatched lengths', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); // Different length
    const close = new Float64Array([7.0, 17.0, 27.0]); // Another different length
    
    assert.throws(
        () => wasm.atr(high, low, close, 14),
        /differing lengths|same length|Inconsistent slice lengths/
    );
});

test('ATR with invalid length', () => {
    const high = new Float64Array(50).fill(100);
    const low = new Float64Array(50).fill(90);
    const close = new Float64Array(50).fill(95);
    
    // Zero length
    assert.throws(
        () => wasm.atr(high, low, close, 0),
        /Invalid length/
    );
});

test('ATR when length exceeds data length', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(
        () => wasm.atr(high, low, close, 10),
        /Not enough data|too short/
    );
});

test('ATR with constant range', () => {
    const length = 50;
    const constantPrice = 100.0;
    const high = new Float64Array(length).fill(constantPrice);
    const low = new Float64Array(length).fill(constantPrice);
    const close = new Float64Array(length).fill(constantPrice);
    
    const result = wasm.atr(high, low, close, 14);
    
    // With constant price (high = low), ATR should be 0 after warmup
    const warmup = 14 - 1;
    for (let i = warmup; i < length; i++) {
        assert.ok(Math.abs(result[i]) < 1e-10, 
            `Expected 0 at index ${i}, got ${result[i]}`);
    }
});

test('ATR in trending market with volatility', () => {
    const length = 100;
    // Create data with increasing volatility
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    
    for (let i = 0; i < length; i++) {
        const price = 100 + i * 0.5; // Uptrending
        const range = 1 + i * 0.1; // Increasing range
        high[i] = price + range;
        low[i] = price - range;
        close[i] = price;
    }
    
    const result = wasm.atr(high, low, close, 14);
    
    // ATR should generally increase with increasing volatility
    const early = result.slice(20, 30); // After warmup
    const late = result.slice(-10);
    
    const earlyAvg = early.reduce((a, b) => a + b) / early.length;
    const lateAvg = late.reduce((a, b) => a + b) / late.length;
    
    assert.ok(lateAvg > earlyAvg, 'ATR should increase with increasing volatility');
});

test('ATR batch calculation with single parameters', () => {
    const { high, low, close } = testData;
    
    const result = wasm.atrBatch(
        high, low, close,
        14, 14, 0  // length range (single value)
    );
    
    // Batch returns flat array
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    // Should match single calculation
    const singleResult = wasm.atr(high, low, close, 14);
    assertArrayClose(
        Array.from(result), 
        Array.from(singleResult), 
        1e-10,
        'Batch vs single calculation mismatch'
    );
});

test('ATR batch calculation with parameter sweep', () => {
    const { high, low, close } = testData;
    const dataLen = Math.min(close.length, 100); // Use smaller subset for speed
    const highSubset = high.slice(0, dataLen);
    const lowSubset = low.slice(0, dataLen);
    const closeSubset = close.slice(0, dataLen);
    
    const result = wasm.atrBatch(
        highSubset, lowSubset, closeSubset,
        10, 20, 5  // lengths: 10, 15, 20
    );
    
    // Should have 3 combinations
    const expectedRows = 3;
    assert.strictEqual(result.length, expectedRows * dataLen);
});

test('ATR batch metadata', () => {
    // For length 10-20 step 5
    const meta = wasm.atrBatchMetadata(10, 20, 5);
    
    assert.ok(meta instanceof Float64Array || Array.isArray(meta));
    assert.strictEqual(meta.length, 3); // 10, 15, 20
    
    // Check values
    assert.strictEqual(meta[0], 10);
    assert.strictEqual(meta[1], 15);
    assert.strictEqual(meta[2], 20);
});

test('ATR batch (unified API)', () => {
    const { high, low, close } = testData;
    
    const config = {
        length_range: [14, 14, 0]
    };
    
    const result = wasm.atr_batch(high, low, close, config);
    
    assert.ok(result);
    assert.ok(result.values);
    assert.ok(result.combos);
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, close.length);
    
    // Check combo structure
    assert.strictEqual(result.combos[0].length, 14);
});

test('ATR error handling coverage', () => {
    const { high, low, close } = testData;
    
    // InvalidLength
    assert.throws(
        () => wasm.atr(high.slice(0, 50), low.slice(0, 50), close.slice(0, 50), 0),
        /Invalid length/
    );
    
    // InconsistentSliceLengths
    assert.throws(
        () => wasm.atr(high.slice(0, 50), low.slice(0, 49), close.slice(0, 50), 14),
        /differing lengths|Inconsistent slice lengths/
    );
    
    // NoCandlesAvailable
    assert.throws(
        () => wasm.atr(new Float64Array([]), new Float64Array([]), new Float64Array([]), 14),
        /No candles|no data/
    );
    
    // NotEnoughData
    assert.throws(
        () => wasm.atr(high.slice(0, 10), low.slice(0, 10), close.slice(0, 10), 20),
        /Not enough data/
    );
});

test('ATR real-world conditions', () => {
    const { high, low, close } = testData;
    
    const result = wasm.atr(high, low, close, 14);
    
    // Check warmup period behavior
    const warmup = 14 - 1; // length - 1
    
    // Values before warmup should be NaN
    for (let i = 0; i < warmup; i++) {
        assert.ok(isNaN(result[i]));
    }
    
    // Values from warmup onwards should not be NaN
    const validStart = warmup;
    for (let i = validStart; i < result.length; i++) {
        assert.ok(!isNaN(result[i]));
    }
    
    // Check output properties
    assert.strictEqual(result.length, close.length);
    
    // ATR should always be non-negative
    const validValues = Array.from(result.slice(validStart));
    assert.ok(validValues.every(v => v >= 0), 'ATR should be non-negative');
    
    // ATR should be positive in markets with volatility
    assert.ok(validValues.some(v => v > 0), 'Should have some positive ATR values');
});

test('ATR with exactly length data points', () => {
    const length = 14;
    const high = new Float64Array(length).fill(110.0);
    const low = new Float64Array(length).fill(90.0);
    const close = new Float64Array(length).fill(100.0);
    
    const result = wasm.atr(high, low, close, length);
    
    assert.strictEqual(result.length, length);
    // First length-1 should be NaN
    for (let i = 0; i < length - 1; i++) {
        assert.ok(isNaN(result[i]));
    }
    // Last value should be valid
    assert.ok(!isNaN(result[length - 1]));
    assert.ok(result[length - 1] >= 0);
});

test('ATR comparison with Rust', () => {
    const { high, low, close } = testData;
    const expected = EXPECTED_OUTPUTS.atr;
    
    const result = wasm.atr(
        high, low, close,
        expected.defaultParams.length
    );
    
    compareWithRust('atr', Array.from(result), 'ohlc', expected.defaultParams);
});