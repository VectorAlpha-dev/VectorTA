/**
 * WASM binding tests for Squeeze Momentum indicator.
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

test('Squeeze Momentum partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    
    const result = wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
    assert.strictEqual(result.length, close.length * 3); 
});

test('Squeeze Momentum accuracy', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    const close = new Float64Array(testData.close);
    const expectedLastFive = [-170.9, -155.4, -65.3, -61.1, -178.1];
    
    const result = wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
    
    
    const dataLen = close.length;
    const momentum = result.slice(dataLen, dataLen * 2);
    
    assert.strictEqual(momentum.length, close.length);
    
    
    const last5Momentum = momentum.slice(-5);
    for (let i = 0; i < 5; i++) {
        if (isNaN(expectedLastFive[i])) {
            assert(isNaN(last5Momentum[i]), `Expected NaN at index ${i}, got ${last5Momentum[i]}`);
        } else {
            assertClose(last5Momentum[i], expectedLastFive[i], 0.1, 
                       `SMI momentum mismatch at index ${i}`);
        }
    }
});

test('Squeeze Momentum zero length', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([10.0, 20.0, 30.0]);
    const close = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.squeeze_momentum_js(high, low, close, 0, 2.0, 0, 1.5);
    }, /Invalid length/);
});

test('Squeeze Momentum length exceeds', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([10.0, 20.0, 30.0]);
    const close = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.squeeze_momentum_js(high, low, close, 10, 2.0, 10, 1.5);
    }, /Invalid length/);
});

test('Squeeze Momentum all NaN', () => {
    
    const high = new Float64Array([NaN, NaN, NaN]);
    const low = new Float64Array([NaN, NaN, NaN]);
    const close = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
    });
});

test('Squeeze Momentum fast API (in-place)', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    const len = high.length;
    
    
    const input = new Float64Array(len * 3);
    input.set(high, 0);
    input.set(low, len);
    input.set(close, len * 2);
    
    
    const squeeze = wasm.squeeze_momentum_alloc(len);
    const momentum = wasm.squeeze_momentum_alloc(len);
    const momentumSignal = wasm.squeeze_momentum_alloc(len);
    
    
    const inputPtr = wasm.squeeze_momentum_alloc(len * 3);
    const inputView = new Float64Array(wasm.__wasm.memory.buffer, inputPtr, len * 3);
    inputView.set(input);
    
    try {
        
        wasm.squeeze_momentum_into(
            inputPtr,
            squeeze,
            momentum,
            momentumSignal,
            len,
            20, 2.0, 20, 1.5
        );
        
        
        const squeezeResult = new Float64Array(wasm.__wasm.memory.buffer, squeeze, len);
        const momentumResult = new Float64Array(wasm.__wasm.memory.buffer, momentum, len);
        const momentumSignalResult = new Float64Array(wasm.__wasm.memory.buffer, momentumSignal, len);
        
        
        const safeResult = wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
        const safeSqueeze = safeResult.slice(0, len);
        const safeMomentum = safeResult.slice(len, len * 2);
        const safeMomentumSignal = safeResult.slice(len * 2);
        
        assertArrayClose(squeezeResult, safeSqueeze, 1e-10, "Squeeze mismatch");
        assertArrayClose(momentumResult, safeMomentum, 1e-10, "Momentum mismatch");
        assertArrayClose(momentumSignalResult, safeMomentumSignal, 1e-10, "Momentum signal mismatch");
    } finally {
        
        wasm.squeeze_momentum_free(inputPtr, len * 3);
        wasm.squeeze_momentum_free(squeeze, len);
        wasm.squeeze_momentum_free(momentum, len);
        wasm.squeeze_momentum_free(momentumSignal, len);
    }
});

test('Squeeze Momentum batch single param', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    const config = {
        length_bb_range: [20, 20, 0],
        mult_bb_range: [2.0, 2.0, 0.0],
        length_kc_range: [20, 20, 0],
        mult_kc_range: [1.5, 1.5, 0.0]
    };
    
    const result = wasm.squeeze_momentum_batch(high, low, close, config);
    
    
    const single = wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
    const singleMomentum = single.slice(close.length, close.length * 2);
    
    assert.strictEqual(result.rows, 1);
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 100);
    
    assertArrayClose(result.values, singleMomentum, 1e-10, "Batch vs single mismatch");
    assert.deepStrictEqual(result.length_bb, [20]);
    assert.deepStrictEqual(result.mult_bb, [2.0]);
    assert.deepStrictEqual(result.length_kc, [20]);
    assert.deepStrictEqual(result.mult_kc, [1.5]);
});

test('Squeeze Momentum batch multiple params', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const close = new Float64Array(testData.close.slice(0, 50));
    
    const config = {
        length_bb_range: [15, 25, 5],  
        mult_bb_range: [2.0, 2.0, 0.0],  
        length_kc_range: [20, 20, 0],  
        mult_kc_range: [1.0, 2.0, 0.5]  
    };
    
    const result = wasm.squeeze_momentum_batch(high, low, close, config);
    
    
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 9 * 50);
    assert.strictEqual(result.length_bb.length, 9);
    assert.strictEqual(result.mult_bb.length, 9);
    assert.strictEqual(result.length_kc.length, 9);
    assert.strictEqual(result.mult_kc.length, 9);
    
    
    const expectedLengthBb = [15, 15, 15, 20, 20, 20, 25, 25, 25];
    const expectedMultKc = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0];
    
    assert.deepStrictEqual(result.length_bb, expectedLengthBb);
    assertArrayClose(result.mult_kc, expectedMultKc, 1e-10, "mult_kc parameters mismatch");
});

test('Squeeze Momentum edge cases', () => {
    
    const high = new Float64Array(20).fill(100);
    const low = new Float64Array(20).fill(90);
    const close = new Float64Array(20).fill(95);
    
    const result = wasm.squeeze_momentum_js(high, low, close, 20, 2.0, 20, 1.5);
    
    assert.strictEqual(result.length, 20 * 3);
    
    
    const squeeze = result.slice(0, 20);
    const momentum = result.slice(20, 40);
    const momentumSignal = result.slice(40, 60);
    
    
    for (let i = 0; i < 19; i++) {
        assert(isNaN(squeeze[i]), `Expected NaN in squeeze at ${i}`);
        assert(isNaN(momentum[i]), `Expected NaN in momentum at ${i}`);
        assert(isNaN(momentumSignal[i]), `Expected NaN in momentum_signal at ${i}`);
    }
    
    
    assert.throws(() => {
        wasm.squeeze_momentum_js(
            new Float64Array([]),
            new Float64Array([]),
            new Float64Array([]),
            20, 2.0, 20, 1.5
        );
    });
});