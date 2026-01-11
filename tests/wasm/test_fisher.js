/**
 * WASM binding tests for Fisher Transform indicator.
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

test('Fisher partial params', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    
    const result = wasm.fisher_js(high, low, 9);
    assert.strictEqual(result.rows, 2);
    assert.strictEqual(result.cols, high.length);
    assert.strictEqual(result.values.length, high.length * 2);
    
    
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
});

test('Fisher accuracy', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    
    
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
    
    
    const expectedLast5Fisher = [
        -0.4720164683904261,
        -0.23467530106650444,
        -0.14879388501136784,
        -0.026651419122953053,
        -0.2569225042442664,
    ];
    
    
    const last5 = fisher.slice(-5);
    assertArrayClose(
        last5,
        expectedLast5Fisher,
        0.1,  
        "Fisher last 5 values mismatch"
    );
});

test('Fisher zero period', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 0);
    }, /Invalid period/);
});

test('Fisher period exceeds length', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 10);
    }, /Invalid period/);
});

test('Fisher very small dataset', () => {
    
    const high = new Float64Array([10.0]);
    const low = new Float64Array([5.0]);
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 9);
    }, /Invalid period|Not enough valid data/);
});

test('Fisher empty input', () => {
    
    const empty = new Float64Array([]);
    
    assert.throws(() => {
        wasm.fisher_js(empty, empty, 9);
    }, /Empty data/);
});

test('Fisher reinput', () => {
    
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    
    
    const result1 = wasm.fisher_js(high, low, 3);
    const fisher1 = new Float64Array(result1.values.slice(0, high.length));
    const signal1 = new Float64Array(result1.values.slice(high.length));
    
    assert.strictEqual(fisher1.length, high.length);
    assert.strictEqual(signal1.length, high.length);
    
    
    const result2 = wasm.fisher_js(fisher1, signal1, 3);
    const fisher2 = result2.values.slice(0, fisher1.length);
    const signal2 = result2.values.slice(fisher1.length);
    
    assert.strictEqual(fisher2.length, fisher1.length);
    assert.strictEqual(signal2.length, signal1.length);
});

test('Fisher nan handling', () => {
    
    const high = new Float64Array(testData.high);
    const low = new Float64Array(testData.low);
    
    const result = wasm.fisher_js(high, low, 9);
    const fisher = result.values.slice(0, high.length);
    const signal = result.values.slice(high.length);
    
    assert.strictEqual(fisher.length, high.length);
    assert.strictEqual(signal.length, high.length);
    
    
    assertAllNaN(fisher.slice(0, 8), "Expected NaN in warmup period for fisher");
    assertAllNaN(signal.slice(0, 8), "Expected NaN in warmup period for signal");
    
    
    if (fisher.length > 240) {
        assertNoNaN(fisher.slice(240), "Found NaN after warmup in fisher");
        assertNoNaN(signal.slice(240), "Found NaN after warmup in signal");
    }
});

test('Fisher all nan input', () => {
    
    const allNan = new Float64Array(100).fill(NaN);
    
    assert.throws(() => {
        wasm.fisher_js(allNan, allNan, 9);
    }, /All values are NaN/);
});


test('Fisher fast API basic', () => {
    
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    const len = high.length;
    
    
    const highPtr = wasm.fisher_alloc(len);
    const lowPtr = wasm.fisher_alloc(len);
    const outPtr = wasm.fisher_alloc(len * 2);  
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, len);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, len);
        
        
        highView.set(high);
        lowView.set(low);
        
        
        wasm.fisher_into(
            highPtr,
            lowPtr,
            outPtr,  
            len,
            3
        );
        
        
        const output = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len * 2);
        const fisherOut = output.slice(0, len);
        const signalOut = output.slice(len);
        
        
        const safeResult = wasm.fisher_js(high, low, 3);
        const safeFisher = safeResult.values.slice(0, len);
        const safeSignal = safeResult.values.slice(len);
        
        assertArrayClose(fisherOut, safeFisher, 1e-10, "Fast API fisher mismatch");
        assertArrayClose(signalOut, safeSignal, 1e-10, "Fast API signal mismatch");
    } finally {
        
        wasm.fisher_free(highPtr, len);
        wasm.fisher_free(lowPtr, len);
        wasm.fisher_free(outPtr, len * 2);
    }
});

test('Fisher fast API aliasing', () => {
    
    const high = new Float64Array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
    const low = new Float64Array([5.0, 7.0, 9.0, 10.0, 13.0, 15.0]);
    const len = high.length;
    
    
    const highCopy = new Float64Array(high);
    const lowCopy = new Float64Array(low);
    
    
    const dataPtr = wasm.fisher_alloc(len * 2);
    
    try {
        
        let dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len * 2);
        dataView.set(high, 0);
        dataView.set(low, len);
        
        
        wasm.fisher_into(
            dataPtr,       
            dataPtr + len * 8,  
            dataPtr,       
            len,
            3
        );
        
        
        dataView = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len * 2);
        const fisherOut = dataView.slice(0, len);
        const signalOut = dataView.slice(len);
        
        
        const safeResult = wasm.fisher_js(highCopy, lowCopy, 3);
        const safeFisher = safeResult.values.slice(0, len);
        const safeSignal = safeResult.values.slice(len);
        
        assertArrayClose(fisherOut, safeFisher, 1e-10, "Fast API aliased fisher mismatch");
        assertArrayClose(signalOut, safeSignal, 1e-10, "Fast API aliased signal mismatch");
    } finally {
        
        wasm.fisher_free(dataPtr, len * 2);
    }
});


test('Fisher batch single period', () => {
    
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    
    const config = {
        period_range: [9, 9, 1]
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    assert.strictEqual(result.rows, 2);  
    assert.strictEqual(result.cols, 100);
    assert.strictEqual(result.values.length, 2 * 100);  
    assert.strictEqual(result.combos.length, 1);
    assert.strictEqual(result.combos[0].period, 9);
    
    
    const fisher = result.values.slice(0, 100);
    const signal = result.values.slice(100, 200);
    
    
    const singleResult = wasm.fisher_js(high, low, 9);
    const singleFisher = singleResult.values.slice(0, 100);
    const singleSignal = singleResult.values.slice(100);
    
    assertArrayClose(fisher, singleFisher, 1e-10, "Batch fisher mismatch");
    assertArrayClose(signal, singleSignal, 1e-10, "Batch signal mismatch");
});

test('Fisher batch multiple periods', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    
    const config = {
        period_range: [5, 9, 2]  
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    assert.strictEqual(result.rows, 6);  
    assert.strictEqual(result.cols, 50);
    assert.strictEqual(result.values.length, 6 * 50);  
    assert.strictEqual(result.combos.length, 3);
    
    const periods = result.combos.map(c => c.period);
    assert.deepStrictEqual(periods, [5, 7, 9]);
    
    
    for (let i = 0; i < 3; i++) {
        const period = periods[i];
        const rowStart = i * 2 * 50;  
        const fisher = result.values.slice(rowStart, rowStart + 50);
        const signal = result.values.slice(rowStart + 50, rowStart + 100);
        
        const singleResult = wasm.fisher_js(high, low, period);
        const singleFisher = singleResult.values.slice(0, 50);
        const singleSignal = singleResult.values.slice(50);
        
        assertArrayClose(fisher, singleFisher, 1e-10, `Batch fisher period ${period} mismatch`);
        assertArrayClose(signal, singleSignal, 1e-10, `Batch signal period ${period} mismatch`);
    }
});


test('Fisher memory allocation and deallocation', () => {
    
    const len = 1000;
    
    
    const ptr = wasm.fisher_alloc(len);
    assert(ptr !== 0, "Failed to allocate memory");
    
    
    assert.doesNotThrow(() => {
        wasm.fisher_free(ptr, len);
    });
    
    
    assert.doesNotThrow(() => {
        wasm.fisher_free(0, len);
    });
});


test('Fisher mismatched lengths', () => {
    
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]);  
    
    assert.throws(() => {
        wasm.fisher_js(high, low, 2);
    }, /Mismatched data length/);
});

test('Fisher warmup period behavior', () => {
    
    const high = new Float64Array(testData.high.slice(0, 50));
    const low = new Float64Array(testData.low.slice(0, 50));
    const period = 9;
    
    const result = wasm.fisher_js(high, low, period);
    const fisher = result.values.slice(0, 50);
    const signal = result.values.slice(50);
    
    
    let firstValid = -1;
    for (let i = 0; i < fisher.length; i++) {
        if (!isNaN(fisher[i])) {
            firstValid = i;
            break;
        }
    }
    
    
    assert.strictEqual(firstValid, period - 1, `First valid value at wrong index: ${firstValid} vs expected ${period-1}`);
    
    
    for (let i = period; i < fisher.length; i++) {
        assertClose(signal[i], fisher[i-1], 1e-10, `Signal lag property violated at index ${i}`);
    }
});

test('Fisher extreme values', () => {
    
    const high = new Float64Array([1e10, 1e11, 1e12, 1e13, 1e14]);
    const low = new Float64Array([1e9, 1e10, 1e11, 1e12, 1e13]);
    
    const result = wasm.fisher_js(high, low, 3);
    const fisher = result.values.slice(0, 5);
    const signal = result.values.slice(5);
    
    
    for (let i = 2; i < 5; i++) {
        assert(isFinite(fisher[i]), `Fisher produced non-finite value at index ${i}`);
        assert(isFinite(signal[i]), `Signal produced non-finite value at index ${i}`);
    }
});

test('Fisher constant price', () => {
    
    const high = new Float64Array(20).fill(100.0);
    const low = new Float64Array(20).fill(100.0);
    
    const result = wasm.fisher_js(high, low, 5);
    const fisher = result.values.slice(0, 20);
    
    
    
    
    for (let i = 4; i < 20; i++) {  
        assert(!isNaN(fisher[i]), `Fisher should not be NaN at index ${i}`);
        assert(isFinite(fisher[i]), `Fisher should be finite at index ${i}`);
    }
});

test('Fisher batch comprehensive sweep', () => {
    
    const high = new Float64Array(testData.high.slice(0, 30));
    const low = new Float64Array(testData.low.slice(0, 30));
    
    const config = {
        period_range: [3, 9, 3]  
    };
    
    const result = wasm.fisher_batch(high, low, config);
    
    
    assert.strictEqual(result.rows, 6);
    assert.strictEqual(result.combos.length, 3);
    
    const periods = [3, 6, 9];
    
    
    for (let i = 0; i < 3; i++) {
        const period = periods[i];
        const rowStart = i * 2 * 30;
        const fisher = result.values.slice(rowStart, rowStart + 30);
        const signal = result.values.slice(rowStart + 30, rowStart + 60);
        
        
        for (let j = 0; j < period - 1; j++) {
            assert(isNaN(fisher[j]), `Expected NaN at index ${j} for period ${period}`);
            assert(isNaN(signal[j]), `Expected NaN at index ${j} for period ${period}`);
        }
        
        
        assert(!isNaN(fisher[period - 1]), `Expected valid value at index ${period - 1} for period ${period}`);
    }
});