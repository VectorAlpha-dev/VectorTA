/**
 * WASM binding tests for ACOSC indicator.
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
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('ACOSC partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    assert.strictEqual(result.length, high.length * 2); 
});

test('ACOSC accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const expected = EXPECTED_OUTPUTS.acosc;
    
    const result = wasm.acosc_js(high, low);
    
    
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    assert.strictEqual(osc.length, len);
    assert.strictEqual(change.length, len);
    
    
    const last5Osc = osc.slice(-5);
    assertArrayClose(
        last5Osc,
        expected.last5Osc,
        0.1,  
        "ACOSC osc last 5 values mismatch"
    );
    
    
    const last5Change = change.slice(-5);
    assertArrayClose(
        last5Change,
        expected.last5Change,
        0.1,  
        "ACOSC change last 5 values mismatch"
    );
    
    
    
    
});

test('ACOSC too short', () => {
    
    const high = new Float64Array([100.0, 101.0]);
    const low = new Float64Array([99.0, 98.0]);
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Not enough data/);
});

test('ACOSC length mismatch', () => {
    
    const high = new Float64Array([100.0, 101.0, 102.0]);
    const low = new Float64Array([99.0, 98.0]);  
    
    assert.throws(() => {
        wasm.acosc_js(high, low);
    }, /Mismatch/);
});

test('ACOSC NaN handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.acosc_js(high, low);
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    
    assertAllNaN(osc.slice(0, 38), "Expected NaN in warmup period for osc");
    assertAllNaN(change.slice(0, 38), "Expected NaN in warmup period for change");
    
    
    if (osc.length > 240) {
        for (let i = 240; i < osc.length; i++) {
            assert(!isNaN(osc[i]), `Found unexpected NaN in osc at index ${i}`);
            assert(!isNaN(change[i]), `Found unexpected NaN in change at index ${i}`);
        }
    }
});

test('ACOSC leading NaNs', () => {
    
    
    const nanArray = new Float64Array(10);
    nanArray.fill(NaN);
    const dataArray = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        dataArray[i] = 100 + i;
    }
    
    const high = new Float64Array(210);
    high.set(nanArray, 0);
    high.set(dataArray, 10);
    
    const low = new Float64Array(210);
    low.set(nanArray, 0);
    const lowData = new Float64Array(200);
    for (let i = 0; i < 200; i++) {
        lowData[i] = 99 + i;
    }
    low.set(lowData, 10);
    
    const result = wasm.acosc_js(high, low);
    const len = high.length;
    const osc = result.slice(0, len);
    const change = result.slice(len, len * 2);
    
    
    const expectedWarmup = 10 + 38;
    
    
    assertAllNaN(osc.slice(0, expectedWarmup), `Expected NaN in warmup period [0:${expectedWarmup}] for osc`);
    assertAllNaN(change.slice(0, expectedWarmup), `Expected NaN in warmup period [0:${expectedWarmup}] for change`);
    
    
    assert(!isNaN(osc[expectedWarmup]), `Expected valid value at index ${expectedWarmup} for osc`);
    assert(!isNaN(change[expectedWarmup]), `Expected valid value at index ${expectedWarmup} for change`);
});

test('ACOSC all NaN input', () => {
    
    const allNanHigh = new Float64Array(100);
    const allNanLow = new Float64Array(100);
    allNanHigh.fill(NaN);
    allNanLow.fill(NaN);
    
    
    assert.throws(() => {
        wasm.acosc_js(allNanHigh, allNanLow);
    }, /Not enough data/);
});

test('ACOSC single point', () => {
    
    const singleHigh = new Float64Array([100.0]);
    const singleLow = new Float64Array([99.0]);
    
    assert.throws(() => {
        wasm.acosc_js(singleHigh, singleLow);
    }, /Not enough data/);
});

test('ACOSC batch single result', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const batchResult = wasm.acosc_batch_js(high, low);
    const singleResult = wasm.acosc_js(high, low);
    
    assert.strictEqual(batchResult.length, singleResult.length);
    assertArrayClose(batchResult, singleResult, 1e-10, "Batch vs single mismatch");
});

test('ACOSC batch metadata', () => {
    
    const metadata = wasm.acosc_batch_metadata_js();
    
    
    assert.strictEqual(metadata.length, 0);
});

test('ACOSC edge cases', () => {
    
    const high = new Float64Array(testData.high.slice(0, 39));
    const low = new Float64Array(testData.low.slice(0, 39));
    
    const result = wasm.acosc_js(high, low);
    assert.strictEqual(result.length, 78); 
    
    const osc = result.slice(0, 39);
    const change = result.slice(39, 78);
    
    
    assertAllNaN(osc.slice(0, 38), "Expected NaN in first 38 values for osc");
    assertAllNaN(change.slice(0, 38), "Expected NaN in first 38 values for change");
    
    
    assert(!isNaN(osc[38]), "Expected valid value at index 38 for osc");
    assert(!isNaN(change[38]), "Expected valid value at index 38 for change");
    
    
    const tooSmallHigh = new Float64Array(testData.high.slice(0, 38));
    const tooSmallLow = new Float64Array(testData.low.slice(0, 38));
    
    assert.throws(() => {
        wasm.acosc_js(tooSmallHigh, tooSmallLow);
    }, /Not enough data/);
    
    
    assert.throws(() => {
        wasm.acosc_js(new Float64Array([]), new Float64Array([]));
    }, /Empty input data|Not enough data/);
});

test('ACOSC batch returns same as single', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const singleResult = wasm.acosc_js(high, low);
    const batchResult = wasm.acosc_batch_js(high, low);
    
    
    assert.strictEqual(singleResult.length, batchResult.length);
    assertArrayClose(singleResult, batchResult, 1e-10, "Batch should match single result");
});

test.after(() => {
    console.log('ACOSC WASM tests completed');
});
