// Batch and zero-copy tests for QQE
// These are added as a separate file to avoid conflicts

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
    // Load WASM module
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

// Batch functions not yet implemented in WASM bindings
test.skip('QQE batch single parameter set', () => {
    // Test batch with single parameter combination
    const close = new Float64Array(testData.close.slice(0, 100));  // Use smaller dataset for speed
    const expected = EXPECTED_OUTPUTS.qqe;
    
    // Using the new batch API
    const batchResult = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [14, 14, 0],
        smoothing_factor_range: [5, 5, 0],
        fast_factor_range: [4.236, 4.236, 0]
    });
    
    // Check structure
    assert.ok(batchResult.fast_values, 'Should have fast_values');
    assert.ok(batchResult.slow_values, 'Should have slow_values');
    assert.ok(batchResult.combos, 'Should have combos');
    assert.strictEqual(batchResult.rows, 1, 'Should have 1 row');
    assert.strictEqual(batchResult.cols, 100, 'Should have 100 columns');
    
    // Should match single calculation
    const singleResult = wasm.qqe_js(close, 14, 5, 4.236);
    const singleFast = singleResult.values.slice(0, singleResult.cols);
    const singleSlow = singleResult.values.slice(singleResult.cols, singleResult.cols * 2);
    
    assertArrayClose(batchResult.fast_values, singleFast, 1e-10, "Batch vs single fast mismatch");
    assertArrayClose(batchResult.slow_values, singleSlow, 1e-10, "Batch vs single slow mismatch");
});

test.skip('QQE batch multiple parameters', () => {
    // Test batch with multiple parameter combinations
    const close = new Float64Array(testData.close.slice(0, 50));  // Use smaller dataset for speed
    
    const batchResult = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [10, 14, 2],      // 10, 12, 14
        smoothing_factor_range: [3, 5, 2],  // 3, 5
        fast_factor_range: [3.0, 4.0, 1.0]  // 3.0, 4.0
    });
    
    // Should have 3 * 2 * 2 = 12 combinations
    assert.strictEqual(batchResult.combos.length, 12);
    assert.strictEqual(batchResult.rows, 12);
    assert.strictEqual(batchResult.cols, 50);
    assert.strictEqual(batchResult.fast_values.length, 12 * 50);
    assert.strictEqual(batchResult.slow_values.length, 12 * 50);
    
    // Check first combination parameters
    assert.strictEqual(batchResult.combos[0].rsi_period, 10);
    assert.strictEqual(batchResult.combos[0].smoothing_factor, 3);
    assertClose(batchResult.combos[0].fast_factor, 3.0, 1e-10, "fast_factor mismatch");
    
    // Each combination should have different results
    const firstRowFast = batchResult.fast_values.slice(0, 50);
    const secondRowFast = batchResult.fast_values.slice(50, 100);
    
    let hasDifference = false;
    for (let i = 30; i < 50; i++) {  // Check after warmup
        if (!isNaN(firstRowFast[i]) && !isNaN(secondRowFast[i])) {
            if (Math.abs(firstRowFast[i] - secondRowFast[i]) > 1e-10) {
                hasDifference = true;
                break;
            }
        }
    }
    assert.ok(hasDifference, 'Different parameters should produce different results');
});

test.skip('QQE batch edge cases', () => {
    // Test edge cases for batch processing
    const close = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    
    // Single value sweep
    const singleBatch = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [5, 5, 1],
        smoothing_factor_range: [3, 3, 0],
        fast_factor_range: [3.0, 3.0, 0]
    });
    
    assert.strictEqual(singleBatch.combos.length, 1);
    assert.strictEqual(singleBatch.fast_values.length, 15);
    assert.strictEqual(singleBatch.slow_values.length, 15);
    
    // Step larger than range
    const largeBatch = wasm.qqe_batch_unified_js(close, {
        rsi_period_range: [5, 7, 10],  // Step larger than range
        smoothing_factor_range: [3, 3, 0],
        fast_factor_range: [3.0, 3.0, 0]
    });
    
    // Should only have rsi_period=5
    assert.strictEqual(largeBatch.combos.length, 1);
    assert.strictEqual(largeBatch.combos[0].rsi_period, 5);
    
    // Empty data should throw
    assert.throws(() => {
        wasm.qqe_batch_unified_js(new Float64Array([]), {
            rsi_period_range: [14, 14, 0],
            smoothing_factor_range: [5, 5, 0],
            fast_factor_range: [4.236, 4.236, 0]
        });
    }, /[Ee]mpty/);
});

test('QQE zero-copy API', () => {
    // Test zero-copy API with in-place computation
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]);
    const len = data.length;
    
    // Allocate buffer for input and output
    const inPtr = wasm.qqe_alloc(len);
    const outPtr = wasm.qqe_alloc(len * 2);  // Need 2x space for fast and slow
    assert(inPtr !== 0, 'Failed to allocate input memory');
    assert(outPtr !== 0, 'Failed to allocate output memory');
    
    try {
        // Get memory view and copy data
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr / 8, len);
        inView.set(data);
        
        // Compute QQE into output buffer
        wasm.qqe_into(inPtr, outPtr, len, 14, 5, 4.236);
        
        // Read results (interleaved fast and slow)
        const memory2 = wasm.__wasm.memory.buffer;  // Re-get in case it grew
        const outView = new Float64Array(memory2, outPtr / 8, len * 2);
        
        // Verify results match regular API
        const regularResult = wasm.qqe_js(data, 14, 5, 4.236);
        const regularFast = regularResult.values.slice(0, regularResult.cols);
        const regularSlow = regularResult.values.slice(regularResult.cols, regularResult.cols * 2);
        
        // Extract fast and slow from interleaved output
        const fast = [];
        const slow = [];
        for (let i = 0; i < len; i++) {
            fast.push(outView[i * 2]);
            slow.push(outView[i * 2 + 1]);
        }
        
        // Compare results
        for (let i = 0; i < len; i++) {
            if (isNaN(regularFast[i]) && isNaN(fast[i])) continue;
            if (isNaN(regularSlow[i]) && isNaN(slow[i])) continue;
            
            assertClose(fast[i], regularFast[i], 1e-10, 
                       `Zero-copy fast mismatch at index ${i}`);
            assertClose(slow[i], regularSlow[i], 1e-10,
                       `Zero-copy slow mismatch at index ${i}`);
        }
    } finally {
        // Always free memory
        wasm.qqe_free(inPtr, len);
        wasm.qqe_free(outPtr, len * 2);
    }
});

test('QQE zero-copy with large dataset', () => {
    // Test with larger dataset to verify memory handling
    const size = 10000;
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = 50.0 + Math.sin(i * 0.01) * 10.0 + Math.random() * 2.0;
    }
    
    const inPtr = wasm.qqe_alloc(size);
    const outPtr = wasm.qqe_alloc(size * 2);
    assert(inPtr !== 0, 'Failed to allocate large input buffer');
    assert(outPtr !== 0, 'Failed to allocate large output buffer');
    
    try {
        const memory = wasm.__wasm.memory.buffer;
        const inView = new Float64Array(memory, inPtr / 8, size);
        inView.set(data);
        
        wasm.qqe_into(inPtr, outPtr, size, 14, 5, 4.236);
        
        // Recreate view in case memory grew
        const memory2 = wasm.__wasm.memory.buffer;
        const outView = new Float64Array(memory2, outPtr / 8, size * 2);
        
        // Check warmup period has NaN
        const warmup = EXPECTED_OUTPUTS.qqe.warmupPeriod;
        for (let i = 0; i < warmup; i++) {
            assert(isNaN(outView[i * 2]), `Expected NaN in fast at warmup index ${i}`);
            assert(isNaN(outView[i * 2 + 1]), `Expected NaN in slow at warmup index ${i}`);
        }
        
        // Check after warmup has values
        for (let i = warmup; i < Math.min(warmup + 100, size); i++) {
            assert(!isNaN(outView[i * 2]), `Unexpected NaN in fast at index ${i}`);
            assert(!isNaN(outView[i * 2 + 1]), `Unexpected NaN in slow at index ${i}`);
        }
    } finally {
        wasm.qqe_free(inPtr, size);
        wasm.qqe_free(outPtr, size * 2);
    }
});

test('QQE zero-copy error handling', () => {
    // Test null pointer
    assert.throws(() => {
        wasm.qqe_into(0, 0, 10, 14, 5, 4.236);
    }, /null pointer|invalid memory/i);
    
    // Test invalid parameters with allocated memory
    const ptr = wasm.qqe_alloc(20);
    const outPtr = wasm.qqe_alloc(40);
    try {
        // Invalid rsi_period
        assert.throws(() => {
            wasm.qqe_into(ptr, outPtr, 20, 0, 5, 4.236);
        }, /Invalid period/);
        
        // Invalid smoothing_factor
        assert.throws(() => {
            wasm.qqe_into(ptr, outPtr, 20, 14, 0, 4.236);
        }, /Invalid|smoothing/);
    } finally {
        wasm.qqe_free(ptr, 20);
        wasm.qqe_free(outPtr, 40);
    }
});

test('QQE memory leak prevention', () => {
    // Allocate and free multiple times to ensure no leaks
    const sizes = [100, 1000, 5000];
    
    for (const size of sizes) {
        const ptr = wasm.qqe_alloc(size);
        assert(ptr !== 0, `Failed to allocate ${size} elements`);
        
        // Write pattern to verify memory
        const memory = wasm.__wasm.memory.buffer;
        const memView = new Float64Array(memory, ptr / 8, size);
        for (let i = 0; i < Math.min(10, size); i++) {
            memView[i] = i * 2.5;
        }
        
        // Verify pattern
        for (let i = 0; i < Math.min(10, size); i++) {
            assert.strictEqual(memView[i], i * 2.5, `Memory corruption at index ${i}`);
        }
        
        // Free memory
        wasm.qqe_free(ptr, size);
    }
});

test.after(() => {
    console.log('QQE WASM batch and zero-copy tests completed');
});