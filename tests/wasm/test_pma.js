/**
 * WASM binding tests for PMA (Predictive Moving Average) indicator.
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

test('PMA default parameters', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.pma_js(close);
    
    
    assert(result, 'Should have result array');
    assert.strictEqual(result.length, close.length * 2, 'Should have predict and trigger arrays concatenated');
    
    
    const predict = result.slice(0, close.length);
    const trigger = result.slice(close.length);
    
    assert.strictEqual(predict.length, close.length);
    assert.strictEqual(trigger.length, close.length);
});

test('PMA accuracy', async () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const hl2 = new Float64Array(high.length);
    
    
    for (let i = 0; i < high.length; i++) {
        hl2[i] = (high[i] + low[i]) / 2;
    }
    
    const result = wasm.pma_js(hl2);
    const predict = result.slice(0, hl2.length);
    const trigger = result.slice(hl2.length);
    
    assert.strictEqual(predict.length, hl2.length);
    assert.strictEqual(trigger.length, hl2.length);
    
    
    const expectedPredict = [
        59208.18749999999,
        59233.83609693878,
        59213.19132653061,
        59199.002551020414,
        58993.318877551,
    ];
    const expectedTrigger = [
        59157.70790816327,
        59208.60076530612,
        59218.6763392857,
        59211.1443877551,
        59123.05019132652,
    ];
    
    
    const last5Predict = Array.from(predict.slice(-5));
    const last5Trigger = Array.from(trigger.slice(-5));
    
    assertArrayClose(
        last5Predict,
        expectedPredict,
        1e-1,
        "PMA predict last 5 values mismatch"
    );
    assertArrayClose(
        last5Trigger,
        expectedTrigger,
        1e-1,
        "PMA trigger last 5 values mismatch"
    );
});

test('PMA with slice', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]);
    
    const result = wasm.pma_js(data);
    const predict = result.slice(0, data.length);
    const trigger = result.slice(data.length);
    
    assert.strictEqual(predict.length, data.length);
    assert.strictEqual(trigger.length, data.length);
});

test('PMA not enough data', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.pma_js(data);
    }, /Not enough valid data|needed = 7/);
});

test('PMA all values NaN', () => {
    
    const data = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.pma_js(data);
    }, /All values are NaN/);
});

test('PMA empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.pma_js(empty);
    }, /empty/i);
});

test('PMA NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.pma_js(close);
    const predict = result.slice(0, close.length);
    const trigger = result.slice(close.length);
    
    assert.strictEqual(predict.length, close.length);
    assert.strictEqual(trigger.length, close.length);
    
    
    
    let firstValid = 0;
    for (let i = 0; i < close.length; i++) {
        if (!isNaN(close[i])) {
            firstValid = i;
            break;
        }
    }
    
    const expectedWarmup = firstValid + 6;
    
    
    for (let i = 0; i < Math.min(expectedWarmup, close.length); i++) {
        assert(isNaN(predict[i]), `Expected NaN in predict warmup at index ${i}`);
        assert(isNaN(trigger[i]), `Expected NaN in trigger warmup at index ${i}`);
    }
});

test('PMA fast API', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.pma_alloc(len);
    const predictPtr = wasm.pma_alloc(len);
    const triggerPtr = wasm.pma_alloc(len);
    
    try {
        
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(close);
        
        
        wasm.pma_into(inPtr, predictPtr, triggerPtr, len);
        
        
        const predictView = new Float64Array(wasm.__wasm.memory.buffer, predictPtr, len);
        const triggerView = new Float64Array(wasm.__wasm.memory.buffer, triggerPtr, len);
        
        const predict = Array.from(predictView);
        const trigger = Array.from(triggerView);
        
        
        const safeResult = wasm.pma_js(close);
        const safePredict = Array.from(safeResult.slice(0, len));
        const safeTrigger = Array.from(safeResult.slice(len));
        
        assertArrayClose(predict, safePredict, 1e-10, "Fast API predict mismatch");
        assertArrayClose(trigger, safeTrigger, 1e-10, "Fast API trigger mismatch");
        
    } finally {
        
        wasm.pma_free(inPtr, len);
        wasm.pma_free(predictPtr, len);
        wasm.pma_free(triggerPtr, len);
    }
});

test('PMA streaming', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const stream = new wasm.PmaStreamWasm();
    const streamResults = [];
    
    
    for (let i = 0; i < Math.min(20, close.length); i++) {
        const result = stream.update(close[i]);
        streamResults.push(result);
    }
    
    
    for (let i = 0; i < 6; i++) {
        assert(isNaN(streamResults[i][0]), `Expected NaN predict during warmup at index ${i}`);
        assert(isNaN(streamResults[i][1]), `Expected NaN trigger during warmup at index ${i}`);
    }
    
    
    for (let i = 6; i < 9 && i < streamResults.length; i++) {
        const [predict, trigger] = streamResults[i];
        assert(!isNaN(predict), `Expected valid predict value after warmup at index ${i}`);
        assert(isNaN(trigger), `Expected NaN trigger (still warming up) at index ${i}`);
    }
    
    
    for (let i = 9; i < streamResults.length; i++) {
        const [predict, trigger] = streamResults[i];
        assert(!isNaN(predict), `Expected valid predict value at index ${i}`);
        assert(!isNaN(trigger), `Expected valid trigger value at index ${i}`);
    }
});

test('PMA batch API', () => {
    
    const close = new Float64Array(testData.close);
    
    
    const result = wasm.pma_batch(close, {});
    
    assert(result.predict, 'Should have predict output');
    assert(result.trigger, 'Should have trigger output');
    assert.strictEqual(result.rows, 1, 'Should have 1 row (no parameter sweep)');
    assert.strictEqual(result.cols, close.length, 'Should have same columns as input length');
    
    
    const singleResult = wasm.pma_js(close);
    const singlePredict = singleResult.slice(0, close.length);
    const singleTrigger = singleResult.slice(close.length);
    
    assertArrayClose(
        Array.from(result.predict),
        Array.from(singlePredict),
        1e-10,
        "Batch predict doesn't match single run"
    );
    assertArrayClose(
        Array.from(result.trigger),
        Array.from(singleTrigger),
        1e-10,
        "Batch trigger doesn't match single run"
    );
});

test('Rust parity', async () => {
    
    const close = new Float64Array(testData.close);
    const wasmResult = wasm.pma_js(close);
    
    const predict = wasmResult.slice(0, close.length);
    
    
    await compareWithRust('pma', predict);
});
