/**
 * WASM binding tests for WaveTrend indicator.
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
        console.error('Failed to load WASM module. Run "wasm-pack build --target nodejs -- --features wasm --no-default-features" first');
        throw error;
    }
    
    testData = loadTestData();
});

/**
 * Helper function to extract wavetrend results from flattened array
 */
function extractWavetrendResults(flatResult, len) {
    return {
        wt1: flatResult.slice(0, len),
        wt2: flatResult.slice(len, 2 * len),
        wt_diff: flatResult.slice(2 * len, 3 * len)
    };
}

test('WaveTrend partial params', () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    const result = wasm.wavetrend_js(hlc3, 9, 12, 3, 0.015);
    assert.strictEqual(result.length, hlc3.length * 3);
    
    const { wt1, wt2, wt_diff } = extractWavetrendResults(result, hlc3.length);
    assert.strictEqual(wt1.length, hlc3.length);
    assert.strictEqual(wt2.length, hlc3.length);
    assert.strictEqual(wt_diff.length, hlc3.length);
});

test('WaveTrend accuracy', async () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    const result = wasm.wavetrend_js(hlc3, 9, 12, 3, 0.015);
    const { wt1, wt2, wt_diff } = extractWavetrendResults(result, hlc3.length);
    
    
    const expectedWt1 = [
        -29.02058232514538,
        -28.207769813591664,
        -31.991808642927193,
        -31.9218051759519,
        -44.956245952893866,
    ];
    const expectedWt2 = [
        -30.651043230696555,
        -28.686329669808583,
        -29.740053593887932,
        -30.707127877490105,
        -36.2899532572575,
    ];
    
    
    assertArrayClose(
        wt1.slice(-5),
        expectedWt1,
        1e-6,
        "WaveTrend WT1 last 5 values mismatch"
    );
    
    assertArrayClose(
        wt2.slice(-5),
        expectedWt2,
        1e-6,
        "WaveTrend WT2 last 5 values mismatch"
    );
    
    
    const expectedDiff = expectedWt2.map((w2, i) => w2 - expectedWt1[i]);
    assertArrayClose(
        wt_diff.slice(-5),
        expectedDiff,
        1e-6,
        "WaveTrend WT_DIFF last 5 values mismatch"
    );
});

test('WaveTrend default candles', () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    
    const result = wasm.wavetrend_js(hlc3, 9, 12, 3, 0.015);
    const { wt1, wt2, wt_diff } = extractWavetrendResults(result, hlc3.length);
    
    assert.strictEqual(wt1.length, hlc3.length);
    assert.strictEqual(wt2.length, hlc3.length);
    assert.strictEqual(wt_diff.length, hlc3.length);
});

test('WaveTrend zero channel', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wavetrend_js(inputData, 0, 12, 3, 0.015);
    }, /Invalid channel_length/);
});

test('WaveTrend channel exceeds length', () => {
    
    const dataSmall = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.wavetrend_js(dataSmall, 10, 12, 3, 0.015);
    }, /Invalid channel_length/);
});

test('WaveTrend very small dataset', () => {
    
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.wavetrend_js(singlePoint, 9, 12, 3, 0.015);
    }, /Invalid|Not enough valid data/);
});

test('WaveTrend empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.wavetrend_js(empty, 9, 12, 3, 0.015);
    }, /Empty data/);
});

test('WaveTrend all NaN input', () => {
    
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.wavetrend_js(allNaN, 9, 12, 3, 0.015);
    }, /All values are NaN/);
});

test('WaveTrend NaN handling', () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    const result = wasm.wavetrend_js(hlc3, 9, 12, 3, 0.015);
    const { wt1, wt2, wt_diff } = extractWavetrendResults(result, hlc3.length);
    
    
    if (wt1.length > 240) {
        const nonNanAfterWarmup = wt1.slice(240).filter(v => !isNaN(v));
        assert.strictEqual(nonNanAfterWarmup.length, wt1.length - 240, 
            "Found unexpected NaN after warmup period");
    }
});

test('WaveTrend fast API (in-place)', () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    
    const inPtr = wasm.wavetrend_alloc(hlc3.length);
    assert(inPtr !== 0, 'Failed to allocate input memory');
    
    
    const inView = new Float64Array(
        wasm.__wasm.memory.buffer,
        inPtr,
        hlc3.length
    );
    inView.set(hlc3);
    
    
    const wt1Out = wasm.wavetrend_alloc(hlc3.length);
    const wt2Out = wasm.wavetrend_alloc(hlc3.length);
    const wtDiffOut = wasm.wavetrend_alloc(hlc3.length);
    
    try {
        
        wasm.wavetrend_into(
            inPtr,
            wt1Out,
            wt2Out,
            wtDiffOut,
            hlc3.length,
            9, 12, 3, 0.015
        );
        
        
        
        assert.ok(true, "Fast API executed without error");
    } finally {
        
        wasm.wavetrend_free(inPtr, hlc3.length);
        wasm.wavetrend_free(wt1Out, hlc3.length);
        wasm.wavetrend_free(wt2Out, hlc3.length);
        wasm.wavetrend_free(wtDiffOut, hlc3.length);
    }
});

test('WaveTrend batch operation', () => {
    
    const hlc3 = new Float64Array(testData.close.slice(0, 100).map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    const config = {
        channel_length_range: [9, 11, 2],      
        average_length_range: [12, 13, 1],     
        ma_length_range: [3, 3, 0],            
        factor_range: [0.015, 0.020, 0.005]    
    };
    
    const result = wasm.wavetrend_batch(hlc3, config);
    
    
    assert.strictEqual(result.rows, 8);
    assert.strictEqual(result.cols, hlc3.length);
    assert.strictEqual(result.wt1_values.length, 8 * hlc3.length);
    assert.strictEqual(result.wt2_values.length, 8 * hlc3.length);
    assert.strictEqual(result.wt_diff_values.length, 8 * hlc3.length);
    
    
    assert.strictEqual(result.channel_lengths.length, 8);
    assert.strictEqual(result.average_lengths.length, 8);
    assert.strictEqual(result.ma_lengths.length, 8);
    assert.strictEqual(result.factors.length, 8);
});

test('WaveTrend batch single param set', () => {
    
    const hlc3 = new Float64Array(testData.close.map((c, i) => 
        (testData.high[i] + testData.low[i] + c) / 3
    ));
    
    const config = {
        channel_length_range: [9, 9, 0],
        average_length_range: [12, 12, 0],
        ma_length_range: [3, 3, 0],
        factor_range: [0.015, 0.015, 0.0]
    };
    
    const batchResult = wasm.wavetrend_batch(hlc3, config);
    
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, hlc3.length);
    
    
    const singleResult = wasm.wavetrend_js(hlc3, 9, 12, 3, 0.015);
    const { wt1: singleWt1, wt2: singleWt2, wt_diff: singleWtDiff } = 
        extractWavetrendResults(singleResult, hlc3.length);
    
    const batchWt1 = batchResult.wt1_values.slice(0, hlc3.length);
    const batchWt2 = batchResult.wt2_values.slice(0, hlc3.length);
    const batchWtDiff = batchResult.wt_diff_values.slice(0, hlc3.length);
    
    assertArrayClose(batchWt1, singleWt1, 1e-10);
    assertArrayClose(batchWt2, singleWt2, 1e-10);
    assertArrayClose(batchWtDiff, singleWtDiff, 1e-10);
});
