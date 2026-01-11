/**
 * WASM binding tests for PPO indicator.
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

test('PPO partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ppo_js(close, 12, 26, 'sma');
    assert.strictEqual(result.length, close.length);
});

test('PPO accuracy', async () => {
    
    const close = new Float64Array(testData.close);
    const expected = EXPECTED_OUTPUTS.ppo;
    
    const result = wasm.ppo_js(
        close,
        expected.defaultParams.fast_period,
        expected.defaultParams.slow_period,
        expected.defaultParams.ma_type
    );
    
    assert.strictEqual(result.length, close.length);
    
    
    const last5 = result.slice(-5);
    assertArrayClose(
        last5,
        expected.last5Values,
        1e-8,
        "PPO last 5 values mismatch"
    );
    
    
    await compareWithRust('ppo', result, 'close', expected.defaultParams);
});

test('PPO default candles', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ppo_js(close, 12, 26, 'sma');
    assert.strictEqual(result.length, close.length);
});

test('PPO zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ppo_js(inputData, 0, 26, 'sma');
    }, /invalid period/i);
});

test('PPO period exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.ppo_js(dataSmall, 12, 26, 'sma');
    }, /invalid period/i);
});

test('PPO very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.ppo_js(singlePoint, 12, 26, 'sma');
    }, /invalid period/i);
});

test('PPO empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.ppo_js(empty, 12, 26, 'sma');
    }, /empty|all values are nan|invalid period/i);
});

test('PPO NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.ppo_js(close, 12, 26, 'sma');
    assert.strictEqual(result.length, close.length);
    
    
    if (result.length > 30) {
        for (let i = 30; i < result.length; i++) {
            assert(!isNaN(result[i]), `Found unexpected NaN at index ${i}`);
        }
    }
});

test('PPO different MA types', () => {
    
    const close = new Float64Array(testData.close);
    
    const maTypes = ['sma', 'ema', 'wma'];
    const results = {};
    
    for (const maType of maTypes) {
        results[maType] = wasm.ppo_js(close, 12, 26, maType);
        assert.strictEqual(results[maType].length, close.length);
    }
    
    
    assert(!results.sma.every((v, i) => v === results.ema[i]));
    assert(!results.sma.every((v, i) => v === results.wma[i]));
    assert(!results.ema.every((v, i) => v === results.wma[i]));
});

test('PPO batch processing', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [12, 12, 0], 
        slow_period_range: [26, 26, 0], 
        ma_type: 'sma'
    };
    
    const result = wasm.ppo_batch(close, config);
    
    assert(result.values);
    assert(result.combos);
    assert.strictEqual(result.rows, 1); 
    assert.strictEqual(result.cols, close.length);
    
    
    assert.strictEqual(result.combos[0].fast_period, 12);
    assert.strictEqual(result.combos[0].slow_period, 26);
    assert.strictEqual(result.combos[0].ma_type, 'sma');
    
    
    const singleResult = wasm.ppo_js(close, 12, 26, 'sma');
    const batchValues = result.values.slice(0, close.length);
    assertArrayClose(
        batchValues,
        singleResult,
        1e-9,
        "PPO batch vs single calculation mismatch"
    );
});

test('PPO batch multiple params', () => {
    
    const close = new Float64Array(testData.close);
    
    const config = {
        fast_period_range: [10, 14, 2], 
        slow_period_range: [24, 28, 2], 
        ma_type: 'ema'
    };
    
    const result = wasm.ppo_batch(close, config);
    
    
    assert.strictEqual(result.rows, 9);
    assert.strictEqual(result.cols, close.length);
    assert.strictEqual(result.combos.length, 9);
    
    
    const expectedCombinations = [
        { fast: 10, slow: 24 }, { fast: 10, slow: 26 }, { fast: 10, slow: 28 },
        { fast: 12, slow: 24 }, { fast: 12, slow: 26 }, { fast: 12, slow: 28 },
        { fast: 14, slow: 24 }, { fast: 14, slow: 26 }, { fast: 14, slow: 28 }
    ];
    
    for (let i = 0; i < expectedCombinations.length; i++) {
        assert.strictEqual(result.combos[i].fast_period, expectedCombinations[i].fast);
        assert.strictEqual(result.combos[i].slow_period, expectedCombinations[i].slow);
        assert.strictEqual(result.combos[i].ma_type, 'ema');
    }
});

test('PPO fast API - basic', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const inPtr = wasm.ppo_alloc(len);
    const outPtr = wasm.ppo_alloc(len);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        wasmMemory.set(close, inPtr / 8);
        
        
        wasm.ppo_into(inPtr, outPtr, len, 12, 26, 'sma');
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const resultCopy = new Float64Array(result); 
        
        
        const safeResult = wasm.ppo_js(close, 12, 26, 'sma');
        assertArrayClose(resultCopy, safeResult, 1e-9, "Fast API result mismatch");
        
    } finally {
        
        wasm.ppo_free(inPtr, len);
        wasm.ppo_free(outPtr, len);
    }
});

test('PPO fast API - in-place (aliasing)', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const ptr = wasm.ppo_alloc(len);
    
    try {
        
        const wasmMemory = new Float64Array(wasm.__wasm.memory.buffer);
        wasmMemory.set(close, ptr / 8);
        
        
        wasm.ppo_into(ptr, ptr, len, 12, 26, 'sma');
        
        
        const result = new Float64Array(wasm.__wasm.memory.buffer, ptr, len);
        const resultCopy = new Float64Array(result); 
        
        
        const safeResult = wasm.ppo_js(close, 12, 26, 'sma');
        assertArrayClose(resultCopy, safeResult, 1e-9, "In-place result mismatch");
        
    } finally {
        
        wasm.ppo_free(ptr, len);
    }
});


if (process.argv[1] === fileURLToPath(import.meta.url)) {
    test.run();
}
