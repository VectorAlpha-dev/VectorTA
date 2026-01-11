/**
 * WASM binding tests for NVI indicator.
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

test('NVI accuracy - safe API', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    
    const result = wasm.nvi_js(close, volume);
    
    
    const expected = [
        154243.6925373456,
        153973.11239019397,
        153973.11239019397,
        154275.63921207888,
        154275.63921207888,
    ];
    const last5 = result.slice(-5);
    
    assertArrayClose(last5, expected, 1e-5, 'NVI accuracy test failed');
});

test('NVI error handling - empty data', () => {
    const emptyArray = new Float64Array(0);
    
    assert.throws(() => {
        wasm.nvi_js(emptyArray, emptyArray);
    }, /empty/i, 'Should throw error for empty data');
});

test('NVI error handling - all NaN values', () => {
    const nanClose = new Float64Array([NaN, NaN, NaN]);
    const nanVolume = new Float64Array([NaN, NaN, NaN]);
    
    assert.throws(() => {
        wasm.nvi_js(nanClose, nanVolume);
    }, /All close values are NaN/i, 'Should throw error when all values are NaN');
});

test('NVI error handling - not enough valid data', () => {
    const close = new Float64Array([NaN, 100.0]);
    const volume = new Float64Array([NaN, 120.0]);
    
    assert.throws(() => {
        wasm.nvi_js(close, volume);
    }, /Not enough valid data/i, 'Should throw error when not enough valid data');
});

test('NVI fast API - basic operation', () => {
    const close = testData.close;
    const volume = testData.volume;
    const len = close.length;
    
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        
        wasm.nvi_into(closePtr, volumePtr, outPtr, len);
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const result = Array.from(memory);
        
        
        const safeResult = wasm.nvi_js(close, volume);
        assertArrayClose(result, safeResult, 1e-9, 'Fast API should match safe API');
    } finally {
        
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI fast API - in-place operation (aliasing)', () => {
    const close = testData.close.slice(0, 100); 
    const volume = testData.volume.slice(0, 100);
    const len = close.length;
    
    
    const expected = wasm.nvi_js(close, volume);
    
    
    const dataPtr = wasm.nvi_alloc(len);
    const memory = new Float64Array(wasm.__wasm.memory.buffer, dataPtr, len);
    memory.set(close);
    
    
    const volumePtr = wasm.nvi_alloc(len);
    const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
    volumeMemory.set(volume);
    
    try {
        
        wasm.nvi_into(dataPtr, volumePtr, dataPtr, len);
        
        
        const result = Array.from(memory);
        assertArrayClose(result, expected, 1e-9, 'In-place operation should produce correct results');
    } finally {
        
        wasm.nvi_free(dataPtr, len);
        wasm.nvi_free(volumePtr, len);
    }
});

test('NVI fast API - null pointer handling', () => {
    assert.throws(() => {
        wasm.nvi_into(0, 0, 0, 100);
    }, /null pointer/i, 'Should throw error for null pointers');
});

test('NVI memory management - no leaks', () => {
    
    const sizes = [100, 1000, 10000];
    
    for (const size of sizes) {
        const closePtrs = [];
        const volumePtrs = [];
        const outPtrs = [];
        
        
        for (let i = 0; i < 10; i++) {
            closePtrs.push(wasm.nvi_alloc(size));
            volumePtrs.push(wasm.nvi_alloc(size));
            outPtrs.push(wasm.nvi_alloc(size));
        }
        
        
        for (let i = 0; i < 10; i++) {
            assert(closePtrs[i] !== 0, `Failed to allocate close buffer ${i}`);
            assert(volumePtrs[i] !== 0, `Failed to allocate volume buffer ${i}`);
            assert(outPtrs[i] !== 0, `Failed to allocate output buffer ${i}`);
        }
        
        
        for (let i = 0; i < 10; i++) {
            wasm.nvi_free(closePtrs[i], size);
            wasm.nvi_free(volumePtrs[i], size);
            wasm.nvi_free(outPtrs[i], size);
        }
    }
    
    
});



test('NVI batch - basic single row', () => {
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    const len = close.length;
    
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len); 
    
    try {
        
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        
        const rows = wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        assert.strictEqual(rows, 1, 'Should return 1 row');
        
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const batchResult = Array.from(memory);
        
        
        const regularResult = wasm.nvi_js(close, volume);
        assertArrayClose(batchResult, regularResult, 1e-9, 'Batch result should match regular API');
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - error handling with null pointers', () => {
    assert.throws(() => {
        wasm.nvi_batch_into(0, 0, 0, 100);
    }, /null pointer/i, 'Should throw error for null pointers');
});

test('NVI batch - length mismatch detection', () => {
    const len = 100;
    
    
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.fill(NaN);
        volumeMemory.fill(NaN);
        
        assert.throws(() => {
            wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        }, /all values nan/i, 'Should throw error when all values are NaN');
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - not enough valid data', () => {
    const len = 2;
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        
        
        closeMemory.set([NaN, 100.0]);
        volumeMemory.set([NaN, 120.0]);
        
        assert.throws(() => {
            wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        }, /not enough valid data/i, 'Should throw error when not enough valid data');
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - warmup period handling', () => {
    const close = new Float64Array([NaN, NaN, 100, 101, 102, 103, 104, 105]);
    const volume = new Float64Array([NaN, NaN, 1000, 900, 1100, 800, 1200, 700]);
    const len = close.length;
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        const rows = wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        assert.strictEqual(rows, 1, 'Should return 1 row');
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        assert(isNaN(memory[0]), 'Index 0 should be NaN');
        assert(isNaN(memory[1]), 'Index 1 should be NaN');
        
        
        assert.strictEqual(memory[2], 1000.0, 'NVI should start at 1000.0');
        assert(!isNaN(memory[3]), 'Index 3 should have valid value');
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - large dataset performance', () => {
    const size = 10000;
    const close = new Float64Array(size);
    const volume = new Float64Array(size);
    
    
    for (let i = 0; i < size; i++) {
        close[i] = 100 + Math.sin(i * 0.01) * 10 + Math.random() * 2;
        volume[i] = 1000000 + Math.sin(i * 0.03) * 500000 + Math.random() * 100000;
    }
    
    const closePtr = wasm.nvi_alloc(size);
    const volumePtr = wasm.nvi_alloc(size);
    const outPtr = wasm.nvi_alloc(size);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, size);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, size);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        const startTime = performance.now();
        const rows = wasm.nvi_batch_into(closePtr, volumePtr, outPtr, size);
        const endTime = performance.now();
        
        assert.strictEqual(rows, 1, 'Should return 1 row');
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, size);
        
        
        assert.strictEqual(memory[0], 1000.0, 'NVI should start at 1000.0');
        assert(!isNaN(memory[size - 1]), 'Last value should be valid');
        
        console.log(`NVI batch processed ${size} elements in ${(endTime - startTime).toFixed(2)}ms`);
    } finally {
        wasm.nvi_free(closePtr, size);
        wasm.nvi_free(volumePtr, size);
        wasm.nvi_free(outPtr, size);
    }
});

test('NVI batch - volume decrease pattern verification', () => {
    
    const close = new Float64Array([100, 101, 102, 103, 104, 105]);
    const volume = new Float64Array([1000, 900, 800, 700, 600, 500]); 
    const len = close.length;
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        assert.strictEqual(memory[0], 1000.0, 'NVI starts at 1000.0');
        
        
        let expectedNvi = 1000.0;
        for (let i = 1; i < len; i++) {
            const pctChange = (close[i] - close[i-1]) / close[i-1];
            expectedNvi += expectedNvi * pctChange;
            assertClose(memory[i], expectedNvi, 1e-9, `NVI mismatch at index ${i}`);
        }
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - volume increase pattern verification', () => {
    
    const close = new Float64Array([100, 101, 102, 103, 104, 105]);
    const volume = new Float64Array([1000, 1100, 1200, 1300, 1400, 1500]); 
    const len = close.length;
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        
        
        for (let i = 0; i < len; i++) {
            assert.strictEqual(memory[i], 1000.0, `NVI should stay at 1000.0, got ${memory[i]} at index ${i}`);
        }
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - mixed volume pattern', () => {
    
    const close = new Float64Array([100, 101, 102, 103, 104, 105]);
    const volume = new Float64Array([1000, 900, 1100, 800, 1200, 700]); 
    const len = close.length;
    
    const closePtr = wasm.nvi_alloc(len);
    const volumePtr = wasm.nvi_alloc(len);
    const outPtr = wasm.nvi_alloc(len);
    
    try {
        const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeMemory.set(close);
        volumeMemory.set(volume);
        
        wasm.nvi_batch_into(closePtr, volumePtr, outPtr, len);
        
        const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const regularResult = wasm.nvi_js(close, volume);
        
        
        assertArrayClose(Array.from(memory), regularResult, 1e-9, 'Batch should match regular API for mixed pattern');
    } finally {
        wasm.nvi_free(closePtr, len);
        wasm.nvi_free(volumePtr, len);
        wasm.nvi_free(outPtr, len);
    }
});

test('NVI batch - consistency with safe API across various datasets', () => {
    const testCases = [
        { size: 10, desc: 'small dataset' },
        { size: 100, desc: 'medium dataset' },
        { size: 1000, desc: 'large dataset' }
    ];
    
    for (const testCase of testCases) {
        const close = new Float64Array(testCase.size);
        const volume = new Float64Array(testCase.size);
        
        
        for (let i = 0; i < testCase.size; i++) {
            close[i] = 100 + Math.sin(i * 0.1) * 10;
            volume[i] = 1000000 * (1 + Math.cos(i * 0.15) * 0.5);
        }
        
        const closePtr = wasm.nvi_alloc(testCase.size);
        const volumePtr = wasm.nvi_alloc(testCase.size);
        const outPtr = wasm.nvi_alloc(testCase.size);
        
        try {
            const closeMemory = new Float64Array(wasm.__wasm.memory.buffer, closePtr, testCase.size);
            const volumeMemory = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, testCase.size);
            closeMemory.set(close);
            volumeMemory.set(volume);
            
            wasm.nvi_batch_into(closePtr, volumePtr, outPtr, testCase.size);
            
            const memory = new Float64Array(wasm.__wasm.memory.buffer, outPtr, testCase.size);
            const batchResult = Array.from(memory);
            
            const regularResult = wasm.nvi_js(close, volume);
            
            assertArrayClose(batchResult, regularResult, 1e-9, 
                `Batch should match regular API for ${testCase.desc}`);
        } finally {
            wasm.nvi_free(closePtr, testCase.size);
            wasm.nvi_free(volumePtr, testCase.size);
            wasm.nvi_free(outPtr, testCase.size);
        }
    }
});

test.after(() => {
    console.log('NVI WASM tests completed');
});
