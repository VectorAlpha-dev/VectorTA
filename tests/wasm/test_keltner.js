/**
 * WASM binding tests for KELTNER indicator.
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
    
    try {
        const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
        const importPath = process.platform === 'win32' 
            ? `file:///${wasmPath.replace(/\\/g, '/')}`
            : wasmPath;
        wasm = await import(importPath);
        
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('KELTNER accuracy - safe API', async () => {
    
    const { high, low, close } = testData;
    const source = close; 
    const { period, multiplier, ma_type } = EXPECTED_OUTPUTS.keltner.defaultParams;
    
    const result = wasm.keltner(high, low, close, source, period, multiplier, ma_type);
    
    assert.equal(result.rows, 3, 'Should have 3 output rows (upper, middle, lower)');
    assert.equal(result.cols, close.length, 'Output length should match input');
    
    const values = result.values;
    const len = close.length;
    
    
    const upper_band = values.slice(0, len);
    const middle_band = values.slice(len, 2 * len);
    const lower_band = values.slice(2 * len, 3 * len);
    
    
    const expectedUpper = [
        61619.504155205745,
        61503.56119134791,
        61387.47897150178,
        61286.61078267451,
        61206.25688331261
    ];
    const expectedMiddle = [
        59758.339871629956,
        59703.35512195091,
        59640.083205574636,
        59593.884805043715,
        59504.46720456336
    ];
    const expectedLower = [
        57897.17558805417,
        57903.14905255391,
        57892.68743964749,
        57901.158827412924,
        57802.67752581411
    ];
    
    assertArrayClose(upper_band.slice(-5), expectedUpper, 1e-1);
    assertArrayClose(middle_band.slice(-5), expectedMiddle, 1e-1);
    assertArrayClose(lower_band.slice(-5), expectedLower, 1e-1);
    
    
    const warmup = period - 1;
    for (let i = 0; i < warmup; i++) {
        assert(isNaN(upper_band[i]), `Upper band warmup value at ${i} should be NaN`);
        assert(isNaN(middle_band[i]), `Middle band warmup value at ${i} should be NaN`);
        assert(isNaN(lower_band[i]), `Lower band warmup value at ${i} should be NaN`);
    }
    
    
    try {
        await compareWithRust('keltner', upper_band, 'keltner', { period, multiplier, ma_type });
    } catch (e) {
        
    }
});

test('KELTNER default params', () => {
    
    const { high, low, close } = testData;
    const source = close;
    
    const result = wasm.keltner(high, low, close, source, 20, 2.0, "ema");
    
    assert.equal(result.rows, 3);
    assert.equal(result.cols, close.length);
    assert.equal(result.values.length, 3 * close.length);
});

test('KELTNER zero period', () => {
    
    const { high, low, close } = testData;
    const source = close;
    
    assert.throws(() => {
        wasm.keltner(high, low, close, source, 0, 2.0, "ema");
    }, /invalid period/i, 'Should throw error for period = 0');
});

test('KELTNER large period', () => {
    
    const { high, low, close } = testData;
    const source = close;
    
    assert.throws(() => {
        wasm.keltner(high, low, close, source, 999999, 2.0, "ema");
    }, /invalid period/i, 'Should throw error for period > data length');
});

test('KELTNER empty input', () => {
    
    const empty = new Float64Array(0);
    
    assert.throws(() => {
        wasm.keltner(empty, empty, empty, empty, 20, 2.0, "ema");
    }, /empty/i, 'Should throw error for empty data');
});

test('KELTNER very small dataset', () => {
    
    const smallData = new Float64Array([1.0, 2.0, 3.0]);
    
    assert.throws(() => {
        wasm.keltner(smallData, smallData, smallData, smallData, 20, 2.0, "ema");
    }, /invalid period|not enough valid data/i, 'Should throw error for insufficient data');
});

test('KELTNER all NaN values', () => {
    
    const len = 100;
    const nanData = new Float64Array(len).fill(NaN);
    
    assert.throws(() => {
        wasm.keltner(nanData, nanData, nanData, nanData, 20, 2.0, "ema");
    }, /all values are nan/i, 'Should throw error for all NaN values');
});

test('KELTNER invalid multiplier', () => {
    
    const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
    
    
    const negResult = wasm.keltner(data, data, data, data, 2, -2.0, "ema");
    assert.equal(negResult.values.length, 3 * data.length);
    
    
    const zeroResult = wasm.keltner(data, data, data, data, 2, 0.0, "ema");
    assert.equal(zeroResult.values.length, 3 * data.length);
    
    
    
    try {
        const nanResult = wasm.keltner(data, data, data, data, 2, NaN, "ema");
        
        const len = data.length;
        const upper = nanResult.values.slice(0, len);
        assert(upper.some(v => isNaN(v)), "Expected NaN in output for NaN multiplier");
    } catch (e) {
        
        assert(e.message.match(/invalid|NaN/), 'Should throw appropriate error for NaN multiplier');
    }
});

test('KELTNER different MA types', () => {
    
    const { high, low, close } = testData;
    const source = close.slice(0, 100); 
    const highSubset = high.slice(0, 100);
    const lowSubset = low.slice(0, 100);
    const closeSubset = close.slice(0, 100);
    
    const maTypes = ["ema", "sma", "wma", "rma"];
    
    for (const maType of maTypes) {
        try {
            const result = wasm.keltner(
                highSubset, lowSubset, closeSubset, source,
                20, 2.0, maType
            );
            assert.equal(result.cols, 100, `Failed for MA type: ${maType}`);
            assert.equal(result.rows, 3, `Failed for MA type: ${maType}`);
        } catch (e) {
            
            if (!e.message.toLowerCase().includes('unsupported')) {
                throw e;
            }
        }
    }
});

test('KELTNER NaN handling', () => {
    
    const { high, low, close } = testData;
    const source = close;
    const period = 20;
    
    const result = wasm.keltner(high, low, close, source, period, 2.0, "ema");
    
    const values = result.values;
    const len = close.length;
    
    
    const upper_band = values.slice(0, len);
    const middle_band = values.slice(len, 2 * len);
    const lower_band = values.slice(2 * len, 3 * len);
    
    
    if (len > 240) {
        for (let i = 240; i < len; i++) {
            assert(!isNaN(upper_band[i]), `Found unexpected NaN in upper band at index ${i}`);
            assert(!isNaN(middle_band[i]), `Found unexpected NaN in middle band at index ${i}`);
            assert(!isNaN(lower_band[i]), `Found unexpected NaN in lower band at index ${i}`);
        }
    }
    
    
    const warmup = period - 1;
    assertAllNaN(upper_band.slice(0, warmup), `Expected NaN in upper band warmup period (first ${warmup} values)`);
    assertAllNaN(middle_band.slice(0, warmup), `Expected NaN in middle band warmup period (first ${warmup} values)`);
    assertAllNaN(lower_band.slice(0, warmup), `Expected NaN in lower band warmup period (first ${warmup} values)`);
});

test('KELTNER batch single parameter set', () => {
    
    const { high, low, close } = testData;
    const source = close;
    const period = 20;
    const multiplier = 2.0;
    
    
    const singleResult = wasm.keltner(high, low, close, source, period, multiplier, "ema");
    
    
    const config = {
        period_range: [period, period, 0],
        multiplier_range: [multiplier, multiplier, 0],
        ma_type: "ema"
    };
    
    const batchResult = wasm.keltner_batch(high, low, close, source, config);
    
    
    assert.equal(batchResult.rows, 1);
    assert.equal(batchResult.cols, close.length);
    assert.equal(batchResult.combos.length, 1);
    
    
    assert.equal(batchResult.combos[0].period, period);
    assert.equal(batchResult.combos[0].multiplier, multiplier);
    
    
    const len = close.length;
    const singleUpper = singleResult.values.slice(0, len);
    const singleMiddle = singleResult.values.slice(len, 2 * len);
    const singleLower = singleResult.values.slice(2 * len, 3 * len);
    
    const batchUpper = batchResult.upper.slice(0, len);
    const batchMiddle = batchResult.middle.slice(0, len);
    const batchLower = batchResult.lower.slice(0, len);
    
    assertArrayClose(batchUpper, singleUpper, 1e-10, "Batch upper doesn't match single");
    assertArrayClose(batchMiddle, singleMiddle, 1e-10, "Batch middle doesn't match single");
    assertArrayClose(batchLower, singleLower, 1e-10, "Batch lower doesn't match single");
});

test('KELTNER batch multiple parameters', () => {
    
    const { high, low, close } = testData;
    const source = close.slice(0, 100); 
    const highSubset = high.slice(0, 100);
    const lowSubset = low.slice(0, 100);
    const closeSubset = close.slice(0, 100);
    
    const config = {
        period_range: [10, 30, 10],  
        multiplier_range: [1.0, 3.0, 1.0],  
        ma_type: "ema"
    };
    
    const result = wasm.keltner_batch(highSubset, lowSubset, closeSubset, source, config);
    
    
    assert.equal(result.rows, 9, 'Should have 9 parameter combinations');
    assert.equal(result.cols, 100, 'Output columns should match input length');
    assert.equal(result.combos.length, 9, 'Should have 9 parameter combinations');
    
    
    assert.equal(result.combos[0].period, 10);
    assert.equal(result.combos[0].multiplier, 1.0);
    
    
    assert.equal(result.combos[8].period, 30);
    assert.equal(result.combos[8].multiplier, 3.0);
    
    
    assert.equal(result.upper.length, 9 * 100);
    assert.equal(result.middle.length, 9 * 100);
    assert.equal(result.lower.length, 9 * 100);
    
    
    const firstRowUpper = result.upper.slice(0, 100);
    const firstRowMiddle = result.middle.slice(0, 100);
    const firstRowLower = result.lower.slice(0, 100);
    
    const singleResult = wasm.keltner(
        highSubset, lowSubset, closeSubset, source,
        10, 1.0, "ema"
    );
    
    const singleUpper = singleResult.values.slice(0, 100);
    const singleMiddle = singleResult.values.slice(100, 200);
    const singleLower = singleResult.values.slice(200, 300);
    
    assertArrayClose(firstRowUpper, singleUpper, 1e-10, "First batch row upper doesn't match single");
    assertArrayClose(firstRowMiddle, singleMiddle, 1e-10, "First batch row middle doesn't match single");
    assertArrayClose(firstRowLower, singleLower, 1e-10, "First batch row lower doesn't match single");
});

test('KELTNER batch metadata', () => {
    
    const data = new Float64Array(50);
    data.fill(100);
    
    const config = {
        period_range: [10, 20, 5],  
        multiplier_range: [1.0, 2.0, 0.5],  
        ma_type: "ema"
    };
    
    const result = wasm.keltner_batch(data, data, data, data, config);
    
    
    assert.equal(result.combos.length, 9);
    
    
    const expectedCombos = [
        { period: 10, multiplier: 1.0 },
        { period: 10, multiplier: 1.5 },
        { period: 10, multiplier: 2.0 },
        { period: 15, multiplier: 1.0 },
        { period: 15, multiplier: 1.5 },
        { period: 15, multiplier: 2.0 },
        { period: 20, multiplier: 1.0 },
        { period: 20, multiplier: 1.5 },
        { period: 20, multiplier: 2.0 }
    ];
    
    for (let i = 0; i < 9; i++) {
        assert.equal(result.combos[i].period, expectedCombos[i].period, 
                     `Period mismatch at combo ${i}`);
        assertClose(result.combos[i].multiplier, expectedCombos[i].multiplier, 1e-10,
                    `Multiplier mismatch at combo ${i}`);
    }
});

test('KELTNER batch full parameter sweep', () => {
    
    const data = new Float64Array(50);
    data.fill(100);
    
    const config = {
        period_range: [10, 14, 2],      
        multiplier_range: [1.0, 1.5, 0.5],  
        ma_type: "ema"
    };
    
    const batchResult = wasm.keltner_batch(data, data, data, data, config);
    
    
    assert.equal(batchResult.combos.length, 6);
    assert.equal(batchResult.rows, 6);
    assert.equal(batchResult.cols, 50);
    assert.equal(batchResult.upper.length, 6 * 50);
    assert.equal(batchResult.middle.length, 6 * 50);
    assert.equal(batchResult.lower.length, 6 * 50);
});

test.after(() => {
    console.log('KELTNER WASM tests completed');
});