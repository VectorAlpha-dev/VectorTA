/**
 * WASM binding tests for MACD indicator.
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

test('MACD partial params', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    assert.strictEqual(result.values.length, close.length * 3); 
    assert.strictEqual(result.rows, 3);
    assert.strictEqual(result.cols, close.length);
});

test('MACD accuracy', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    
    
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    
    const expected_macd = [
        -629.8674025082801,
        -600.2986584356258,
        -581.6188884820076,
        -551.1020443476082,
        -560.798510688488,
    ];
    const expected_signal = [
        -721.9744591891067,
        -697.6392990384105,
        -674.4352169271299,
        -649.7685824112256,
        -631.9745680666781,
    ];
    const expected_hist = [
        92.10705668082664,
        97.34064060278467,
        92.81632844512228,
        98.6665380636174,
        71.17605737819008,
    ];
    
    const last5_macd = macd.slice(-5);
    const last5_signal = signal.slice(-5);
    const last5_hist = hist.slice(-5);
    
    
    assertArrayClose(
        last5_macd,
        expected_macd,
        1e-1,
        "MACD last 5 values mismatch"
    );
    assertArrayClose(
        last5_signal,
        expected_signal,
        1e-1,
        "Signal last 5 values mismatch"
    );
    assertArrayClose(
        last5_hist,
        expected_hist,
        1e-1,
        "Histogram last 5 values mismatch"
    );
});

test('MACD zero period', () => {
    
    const inputData = new Float64Array([10.0, 20.0, 30.0]);
    
    
    assert.throws(() => {
        wasm.macd_js(inputData, 0, 26, 9, "ema");
    }, /Invalid period|unreachable/);
});

test('MACD period exceeds length', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 12, 26, 9, "ema");
    }, /Invalid period|unreachable/);
});

test('MACD very small dataset', () => {
    
    const data = new Float64Array([42.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 12, 26, 9, "ema");
    }, /Invalid period|Not enough valid data|unreachable/);
});

test('MACD empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.macd_js(empty, 12, 26, 9, "ema");
    }, /input data slice is empty|invalid period|unreachable/i);
});

test('MACD NaN handling', () => {
    
    const close = new Float64Array(testData.close);
    
    const result = wasm.macd_js(close, 12, 26, 9, "ema");
    
    
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    
    if (close.length > 240) {
        for (let i = 240; i < close.length; i++) {
            assert(!isNaN(macd[i]), `Found unexpected NaN in MACD at index ${i}`);
            assert(!isNaN(signal[i]), `Found unexpected NaN in signal at index ${i}`);
            assert(!isNaN(hist[i]), `Found unexpected NaN in histogram at index ${i}`);
        }
    }
});

test('MACD all NaN input', () => {
    
    const allNaN = new Float64Array(100);
    allNaN.fill(NaN);
    
    assert.throws(() => {
        wasm.macd_js(allNaN, 12, 26, 9, "ema");
    }, /all values are nan/i);
});

test('MACD fast API (in-place)', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const in_ptr = wasm.macd_alloc(len);
    const macd_ptr = wasm.macd_alloc(len);
    const signal_ptr = wasm.macd_alloc(len);
    const hist_ptr = wasm.macd_alloc(len);
    
    try {
        
        const memory = wasm.__wasm.memory;
        const memView = new Float64Array(memory.buffer, in_ptr, len);
        memView.set(close);
        
        
        
        wasm.macd_into(
            in_ptr,
            macd_ptr,
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        
        
        const memory2 = wasm.__wasm.memory;
        const fullView = new Float64Array(memory2.buffer);
        const macd = fullView.subarray(macd_ptr / 8, macd_ptr / 8 + len);
        const signal = fullView.subarray(signal_ptr / 8, signal_ptr / 8 + len);
        const hist = fullView.subarray(hist_ptr / 8, hist_ptr / 8 + len);
        
        
        const macdCopy = Array.from(macd);
        const signalCopy = Array.from(signal);
        const histCopy = Array.from(hist);
        
        
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        const safe_signal = safeResult.values.slice(len, len * 2);
        const safe_hist = safeResult.values.slice(len * 2);
        
        
        assertArrayClose(macdCopy, safe_macd, 1e-10, "Fast API MACD mismatch");
        assertArrayClose(signalCopy, safe_signal, 1e-10, "Fast API signal mismatch");
        assertArrayClose(histCopy, safe_hist, 1e-10, "Fast API histogram mismatch");
    } finally {
        
        wasm.macd_free(in_ptr, len);
        wasm.macd_free(macd_ptr, len);
        wasm.macd_free(signal_ptr, len);
        wasm.macd_free(hist_ptr, len);
    }
});

test('MACD fast API aliasing detection', () => {
    
    const close = new Float64Array(testData.close);
    const len = close.length;
    
    
    const buffer_ptr = wasm.macd_alloc(len);
    const signal_ptr = wasm.macd_alloc(len);
    const hist_ptr = wasm.macd_alloc(len);
    
    try {
        
        const memory = wasm.__wasm.memory;
        const buffer = new Float64Array(memory.buffer, buffer_ptr, len);
        buffer.set(close);
        
        
        
        wasm.macd_into(
            buffer_ptr,
            buffer_ptr, 
            signal_ptr,
            hist_ptr,
            len,
            12, 26, 9, "ema"
        );
        
        
        
        const memory2 = wasm.__wasm.memory;
        const fullView = new Float64Array(memory2.buffer);
        const buffer_after = fullView.subarray(buffer_ptr / 8, buffer_ptr / 8 + len);
        
        
        const bufferCopy = Array.from(buffer_after);
        
        
        const safeResult = wasm.macd_js(close, 12, 26, 9, "ema");
        const safe_macd = safeResult.values.slice(0, len);
        
        assertArrayClose(bufferCopy, safe_macd, 1e-10, "Aliasing handling failed");
    } finally {
        
        wasm.macd_free(buffer_ptr, len);
        wasm.macd_free(signal_ptr, len);
        wasm.macd_free(hist_ptr, len);
    }
});

test('MACD batch single parameter set', () => {
    
    const close = new Float64Array(testData.close);
    
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [12, 12, 0],
        slow_period_range: [26, 26, 0],
        signal_period_range: [9, 9, 0],
        ma_type: "ema"
    });
    
    
    assert.strictEqual(batchResult.rows, 1);
    assert.strictEqual(batchResult.cols, close.length);
    assert.strictEqual(batchResult.values.length, 3 * close.length); 
    
    
    const batch_macd = batchResult.values.slice(0, close.length);
    const batch_signal = batchResult.values.slice(close.length, 2 * close.length);
    const batch_hist = batchResult.values.slice(2 * close.length);
    
    
    const singleResult = wasm.macd_js(close, 12, 26, 9, "ema");
    const single_macd = singleResult.values.slice(0, close.length);
    const single_signal = singleResult.values.slice(close.length, 2 * close.length);
    const single_hist = singleResult.values.slice(2 * close.length);
    
    assertArrayClose(
        batch_macd,
        single_macd,
        1e-10,
        "Batch vs single MACD mismatch"
    );
    assertArrayClose(
        batch_signal,
        single_signal,
        1e-10,
        "Batch vs single signal mismatch"
    );
    assertArrayClose(
        batch_hist,
        single_hist,
        1e-10,
        "Batch vs single histogram mismatch"
    );
});

test('MACD batch multiple parameters', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50)); 
    
    
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [10, 14, 2],   
        slow_period_range: [24, 28, 2],   
        signal_period_range: [8, 10, 1],  
        ma_type: "ema"
    });
    
    
    assert.strictEqual(batchResult.rows, 27);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.values.length, 3 * 27 * 50); 
    assert.strictEqual(batchResult.fast_periods.length, 27);
    assert.strictEqual(batchResult.slow_periods.length, 27);
    assert.strictEqual(batchResult.signal_periods.length, 27);
    
    
    
    const macd_block_size = 27 * 50;
    const firstMacd = batchResult.values.slice(0, 50); 
    
    const singleResult = wasm.macd_js(close, 10, 24, 8, "ema");
    const single_macd = singleResult.values.slice(0, 50);
    
    assertArrayClose(
        firstMacd,
        single_macd,
        1e-10,
        "First batch row should match single calculation"
    );
});

test('MACD unknown MA type', () => {
    
    const data = new Float64Array([10.0, 20.0, 30.0, 40.0, 50.0]);
    
    assert.throws(() => {
        wasm.macd_js(data, 2, 3, 2, "unknown_ma");
    }, /Unknown MA type/);
});

test('MACD warmup periods', () => {
    
    const close = new Float64Array(testData.close);
    const fastPeriod = 12;
    const slowPeriod = 26;
    const signalPeriod = 9;
    
    const result = wasm.macd_js(close, fastPeriod, slowPeriod, signalPeriod, "ema");
    
    
    const macd = result.values.slice(0, close.length);
    const signal = result.values.slice(close.length, close.length * 2);
    const hist = result.values.slice(close.length * 2);
    
    
    const macdWarmup = slowPeriod - 1;
    for (let i = 0; i < macdWarmup; i++) {
        assert(isNaN(macd[i]), `Expected NaN at MACD index ${i} during warmup`);
    }
    
    
    const signalWarmup = slowPeriod + signalPeriod - 2;
    for (let i = 0; i < signalWarmup; i++) {
        assert(isNaN(signal[i]), `Expected NaN at signal index ${i} during warmup`);
        assert(isNaN(hist[i]), `Expected NaN at histogram index ${i} during warmup`);
    }
    
    
    assert(!isNaN(macd[macdWarmup]), `Unexpected NaN at MACD index ${macdWarmup} after warmup`);
    assert(!isNaN(signal[signalWarmup]), `Unexpected NaN at signal index ${signalWarmup} after warmup`);
    assert(!isNaN(hist[signalWarmup]), `Unexpected NaN at histogram index ${signalWarmup} after warmup`);
});

test('MACD different MA types', () => {
    
    const close = new Float64Array(testData.close.slice(0, 100)); 
    
    
    const resultEMA = wasm.macd_js(close, 12, 26, 9, "ema");
    const macdEMA = resultEMA.values.slice(0, close.length);
    assert.strictEqual(macdEMA.length, close.length);
    
    
    const resultSMA = wasm.macd_js(close, 12, 26, 9, "sma");
    const macdSMA = resultSMA.values.slice(0, close.length);
    assert.strictEqual(macdSMA.length, close.length);
    
    
    const resultWMA = wasm.macd_js(close, 12, 26, 9, "wma");
    const macdWMA = resultWMA.values.slice(0, close.length);
    assert.strictEqual(macdWMA.length, close.length);
    
    
    let emaVsSmaMatch = true;
    let emaVsWmaMatch = true;
    for (let i = 50; i < close.length; i++) {
        if (Math.abs(macdEMA[i] - macdSMA[i]) > 1e-5) emaVsSmaMatch = false;
        if (Math.abs(macdEMA[i] - macdWMA[i]) > 1e-5) emaVsWmaMatch = false;
    }
    assert(!emaVsSmaMatch, "EMA and SMA should produce different results");
    assert(!emaVsWmaMatch, "EMA and WMA should produce different results");
});

test('MACD batch edge cases', () => {
    
    const close = new Float64Array(testData.close.slice(0, 50));
    
    
    const largeBatch = wasm.macd_batch(close, {
        fast_period_range: [12, 14, 10], 
        slow_period_range: [26, 26, 0],
        signal_period_range: [9, 9, 0],
        ma_type: "ema"
    });
    
    
    assert.strictEqual(largeBatch.rows, 1);
    assert.strictEqual(largeBatch.fast_periods.length, 1);
    assert.strictEqual(largeBatch.fast_periods[0], 12);
    
    
    const batchResult = wasm.macd_batch(close, {
        fast_period_range: [10, 14, 2],   
        slow_period_range: [24, 28, 2],   
        signal_period_range: [8, 8, 1],   
        ma_type: "ema"
    });
    
    
    assert.strictEqual(batchResult.rows, 9);
    
    
    
    const firstRowMacd = batchResult.values.slice(0, 50);
    const firstRowSignal = batchResult.values.slice(9 * 50, 9 * 50 + 50);
    const firstRowHist = batchResult.values.slice(2 * 9 * 50, 2 * 9 * 50 + 50);
    
    
    for (let i = 0; i < 23; i++) {
        assert(isNaN(firstRowMacd[i]), `Expected NaN in batch MACD at index ${i}`);
    }
    
    
    for (let i = 0; i < 30; i++) {
        assert(isNaN(firstRowSignal[i]), `Expected NaN in batch signal at index ${i}`);
        assert(isNaN(firstRowHist[i]), `Expected NaN in batch histogram at index ${i}`);
    }
    
    
    assert.throws(() => {
        wasm.macd_batch(new Float64Array([]), {
            fast_period_range: [12, 12, 0],
            slow_period_range: [26, 26, 0],
            signal_period_range: [9, 9, 0],
            ma_type: "ema"
        });
    }, /All values are NaN|Input data slice is empty/);
});

test('MACD memory allocation/deallocation', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.macd_alloc(len);
    assert(ptr !== 0, "Allocation should return non-null pointer");
    
    
    assert.doesNotThrow(() => {
        wasm.macd_free(ptr, len);
    });
    
    
    assert.doesNotThrow(() => {
        wasm.macd_free(0, len);
    });
});


test.after(() => {
    if (process.env.RUN_RUST_COMPARISON) {
        compareWithRust('macd', testData);
    }
});
