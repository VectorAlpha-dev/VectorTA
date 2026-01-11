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

test('ATR with default parameters', () => {
    const { high, low, close } = testData;
    const result = wasm.atr(high, low, close, 14);
    
    
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    
    const warmupPeriod = 14 - 1; 
    for (let i = 0; i < warmupPeriod; i++) {
        assert.ok(isNaN(result[i]), `Expected NaN at index ${i} during warmup`);
    }
    
    
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
    
    
    const last5 = Array.from(result.slice(-5));
    assertArrayClose(last5, expected.last5Values, 1e-2, 'ATR last 5 values mismatch');
});

test('ATR with empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.atr(empty, empty, empty, 14),
        /Input data slice is empty|Empty input data|No candles|no data/
    );
});

test('ATR with mismatched lengths', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0]); 
    const close = new Float64Array([7.0, 17.0, 27.0]); 
    
    assert.throws(
        () => wasm.atr(high, low, close, 14),
        /differing lengths|same length|Inconsistent slice lengths/
    );
});

test('ATR with invalid length', () => {
    const high = new Float64Array(50).fill(100);
    const low = new Float64Array(50).fill(90);
    const close = new Float64Array(50).fill(95);
    
    
    assert.throws(
        () => wasm.atr(high, low, close, 0),
        /Invalid length|Invalid period/
    );
});

test('ATR when length exceeds data length', () => {
    const high = new Float64Array([10.0, 20.0, 30.0]);
    const low = new Float64Array([5.0, 15.0, 25.0]);
    const close = new Float64Array([7.0, 17.0, 27.0]);
    
    assert.throws(
        () => wasm.atr(high, low, close, 10),
        /Invalid period|Not enough data|too short/
    );
});

test('ATR with constant range', () => {
    const length = 50;
    const constantPrice = 100.0;
    const high = new Float64Array(length).fill(constantPrice);
    const low = new Float64Array(length).fill(constantPrice);
    const close = new Float64Array(length).fill(constantPrice);
    
    const result = wasm.atr(high, low, close, 14);
    
    
    const warmup = 14 - 1;
    for (let i = warmup; i < length; i++) {
        assert.ok(Math.abs(result[i]) < 1e-10, 
            `Expected 0 at index ${i}, got ${result[i]}`);
    }
});

test('ATR in trending market with volatility', () => {
    const length = 100;
    
    const high = new Float64Array(length);
    const low = new Float64Array(length);
    const close = new Float64Array(length);
    
    for (let i = 0; i < length; i++) {
        const price = 100 + i * 0.5; 
        const range = 1 + i * 0.1; 
        high[i] = price + range;
        low[i] = price - range;
        close[i] = price;
    }
    
    const result = wasm.atr(high, low, close, 14);
    
    
    const early = result.slice(20, 30); 
    const late = result.slice(-10);
    
    const earlyAvg = early.reduce((a, b) => a + b) / early.length;
    const lateAvg = late.reduce((a, b) => a + b) / late.length;
    
    assert.ok(lateAvg > earlyAvg, 'ATR should increase with increasing volatility');
});

test('ATR batch calculation with single parameters', () => {
    const { high, low, close } = testData;
    
    const result = wasm.atrBatch(
        high, low, close,
        14, 14, 0  
    );
    
    
    assert.ok(result instanceof Float64Array || Array.isArray(result));
    assert.strictEqual(result.length, close.length);
    
    
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
    const dataLen = Math.min(close.length, 100); 
    const highSubset = high.slice(0, dataLen);
    const lowSubset = low.slice(0, dataLen);
    const closeSubset = close.slice(0, dataLen);
    
    const result = wasm.atrBatch(
        highSubset, lowSubset, closeSubset,
        10, 20, 5  
    );
    
    
    const expectedRows = 3;
    assert.strictEqual(result.length, expectedRows * dataLen);
});

test('ATR batch metadata', () => {
    
    const meta = wasm.atrBatchMetadata(10, 20, 5);
    
    assert.ok(meta instanceof Float64Array || Array.isArray(meta));
    assert.strictEqual(meta.length, 3); 
    
    
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
    
    
    assert.strictEqual(result.combos[0].length, 14);
});

test('ATR error handling coverage', () => {
    const { high, low, close } = testData;
    
    
    assert.throws(
        () => wasm.atr(high.slice(0, 50), low.slice(0, 50), close.slice(0, 50), 0),
        /Invalid length|Invalid period/
    );
    
    
    assert.throws(
        () => wasm.atr(high.slice(0, 50), low.slice(0, 49), close.slice(0, 50), 14),
        /differing lengths|Inconsistent slice lengths/
    );
    
    
    assert.throws(
        () => wasm.atr(new Float64Array([]), new Float64Array([]), new Float64Array([]), 14),
        /Input data slice is empty|Empty input data|No candles|no data/
    );
    
    
    assert.throws(
        () => wasm.atr(high.slice(0, 10), low.slice(0, 10), close.slice(0, 10), 20),
        /Invalid period|Not enough data/
    );
});

test('ATR real-world conditions', () => {
    const { high, low, close } = testData;
    
    const result = wasm.atr(high, low, close, 14);
    
    
    const warmup = 14 - 1; 
    
    
    for (let i = 0; i < warmup; i++) {
        assert.ok(isNaN(result[i]));
    }
    
    
    const validStart = warmup;
    for (let i = validStart; i < result.length; i++) {
        assert.ok(!isNaN(result[i]));
    }
    
    
    assert.strictEqual(result.length, close.length);
    
    
    const validValues = Array.from(result.slice(validStart));
    assert.ok(validValues.every(v => v >= 0), 'ATR should be non-negative');
    
    
    assert.ok(validValues.some(v => v > 0), 'Should have some positive ATR values');
});

test('ATR with exactly length data points', () => {
    const length = 14;
    const high = new Float64Array(length).fill(110.0);
    const low = new Float64Array(length).fill(90.0);
    const close = new Float64Array(length).fill(100.0);
    
    const result = wasm.atr(high, low, close, length);
    
    assert.strictEqual(result.length, length);
    
    for (let i = 0; i < length - 1; i++) {
        assert.ok(isNaN(result[i]));
    }
    
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


test('ATR zero-copy API', () => {
    const size = 100;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    
    for (let i = 0; i < size; i++) {
        const price = 100 + i * 0.1;
        const range = 2 + Math.sin(i * 0.1);
        high[i] = price + range;
        low[i] = price - range;
        close[i] = price;
    }
    
    const length = 14;
    
    
    const highPtr = wasm.atr_alloc(size);
    const lowPtr = wasm.atr_alloc(size);
    const closePtr = wasm.atr_alloc(size);
    const outPtr = wasm.atr_alloc(size);
    
    assert(highPtr !== 0, 'Failed to allocate high buffer');
    assert(lowPtr !== 0, 'Failed to allocate low buffer');
    assert(closePtr !== 0, 'Failed to allocate close buffer');
    assert(outPtr !== 0, 'Failed to allocate output buffer');
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        
        
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        
        wasm.atr_into(highPtr, lowPtr, closePtr, outPtr, size, length);
        
        
        const regularResult = wasm.atr(high, low, close, length);
        for (let i = 0; i < size; i++) {
            if (isNaN(regularResult[i]) && isNaN(outView[i])) {
                continue; 
            }
            assert(Math.abs(regularResult[i] - outView[i]) < 1e-10,
                   `Zero-copy mismatch at index ${i}: regular=${regularResult[i]}, zerocopy=${outView[i]}`);
        }
        
        
        for (let i = 0; i < length - 1; i++) {
            assert(isNaN(outView[i]), `Expected NaN at warmup index ${i}`);
        }
        
        
        for (let i = length - 1; i < size; i++) {
            assert(!isNaN(outView[i]), `Unexpected NaN at index ${i}`);
            assert(outView[i] >= 0, `ATR should be non-negative at index ${i}`);
        }
    } finally {
        
        wasm.atr_free(highPtr, size);
        wasm.atr_free(lowPtr, size);
        wasm.atr_free(closePtr, size);
        wasm.atr_free(outPtr, size);
    }
});

test('ATR zero-copy error handling', () => {
    
    assert.throws(() => {
        wasm.atr_into(0, 0, 0, 0, 10, 14);
    }, /null pointer|invalid memory/i);
    
    
    const ptr = wasm.atr_alloc(10);
    try {
        
        assert.throws(() => {
            wasm.atr_into(ptr, ptr, ptr, ptr, 10, 0);
        }, /Invalid length|Invalid period/);
    } finally {
        wasm.atr_free(ptr, 10);
    }
});

test('ATR zero-copy batch API', () => {
    const size = 50;
    const high = new Float64Array(size);
    const low = new Float64Array(size);
    const close = new Float64Array(size);
    
    
    for (let i = 0; i < size; i++) {
        const price = 100 + i * 0.1;
        const range = 1 + Math.sin(i * 0.1);
        high[i] = price + range;
        low[i] = price - range;
        close[i] = price;
    }
    
    
    const highPtr = wasm.atr_alloc(size);
    const lowPtr = wasm.atr_alloc(size);
    const closePtr = wasm.atr_alloc(size);
    
    
    const lengthStart = 10;
    const lengthEnd = 18;
    const lengthStep = 4;
    const numCombos = 3; 
    const outSize = numCombos * size;
    const outPtr = wasm.atr_alloc(outSize);
    
    try {
        
        const highView = new Float64Array(wasm.__wasm.memory.buffer, highPtr, size);
        const lowView = new Float64Array(wasm.__wasm.memory.buffer, lowPtr, size);
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        highView.set(high);
        lowView.set(low);
        closeView.set(close);
        
        
        wasm.atr_batch_into(highPtr, lowPtr, closePtr, outPtr, size, 
                           lengthStart, lengthEnd, lengthStep);
        
        
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, outSize);
        
        
        const lengths = [10, 14, 18];
        for (let combo = 0; combo < lengths.length; combo++) {
            const length = lengths[combo];
            const rowStart = combo * size;
            
            
            const regularResult = wasm.atr(high, low, close, length);
            for (let i = 0; i < size; i++) {
                const batchValue = outView[rowStart + i];
                const regularValue = regularResult[i];
                
                if (isNaN(regularValue) && isNaN(batchValue)) {
                    continue; 
                }
                assert(Math.abs(regularValue - batchValue) < 1e-10,
                       `Batch mismatch for length=${length} at index ${i}`);
            }
        }
    } finally {
        wasm.atr_free(highPtr, size);
        wasm.atr_free(lowPtr, size);
        wasm.atr_free(closePtr, size);
        wasm.atr_free(outPtr, outSize);
    }
});


test('ATR streaming context', () => {
    const length = 14;
    const context = new wasm.AtrContext(length);
    
    
    const testData = [
        { high: 110, low: 90, close: 100 },
        { high: 112, low: 92, close: 102 },
        { high: 115, low: 95, close: 105 },
        { high: 113, low: 93, close: 103 },
        { high: 111, low: 91, close: 101 }
    ];
    
    const results = [];
    for (const data of testData) {
        const result = context.update(data.high, data.low, data.close);
        results.push(result);
    }
    
    
    assert(results[0] === null || results[0] === undefined);
    
    
    context.reset();
    
    
    const high = new Float64Array(50);
    const low = new Float64Array(50);
    const close = new Float64Array(50);
    
    for (let i = 0; i < 50; i++) {
        const price = 100 + Math.sin(i * 0.1) * 10;
        const range = 2 + Math.sin(i * 0.2);
        high[i] = price + range;
        low[i] = price - range;
        close[i] = price;
    }
    
    
    const streamResults = [];
    for (let i = 0; i < 50; i++) {
        const result = context.update(high[i], low[i], close[i]);
        streamResults.push(result);
    }
    
    
    for (let i = 0; i < length - 1; i++) {
        assert(streamResults[i] === null || streamResults[i] === undefined,
               `Expected null at index ${i}, got ${streamResults[i]}`);
    }
    
    
    for (let i = length - 1; i < 50; i++) {
        assert(typeof streamResults[i] === 'number' && !isNaN(streamResults[i]),
               `Expected number at index ${i}, got ${streamResults[i]}`);
        assert(streamResults[i] >= 0, `ATR should be non-negative at index ${i}`);
    }
});

test('ATR streaming context error handling', () => {
    
    assert.throws(() => {
        new wasm.AtrContext(0);
    }, /Invalid length|Invalid period/);
});

test('ATR memory management stress test', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const ptrs = [];
        
        
        for (let i = 0; i < 4; i++) { 
            const ptr = wasm.atr_alloc(size);
            assert(ptr !== 0, `Failed to allocate ${size} elements`);
            ptrs.push(ptr);
        }
        
        
        for (let i = 0; i < ptrs.length; i++) {
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptrs[i], size);
            for (let j = 0; j < Math.min(10, size); j++) {
                memView[j] = i * 10 + j;
            }
        }
        
        
        for (let i = 0; i < ptrs.length; i++) {
            const memView = new Float64Array(wasm.__wasm.memory.buffer, ptrs[i], size);
            for (let j = 0; j < Math.min(10, size); j++) {
                assert.strictEqual(memView[j], i * 10 + j, `Memory corruption at buffer ${i}, index ${j}`);
            }
        }
        
        
        for (const ptr of ptrs) {
            wasm.atr_free(ptr, size);
        }
    }
});
