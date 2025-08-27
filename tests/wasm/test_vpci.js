/**
 * WASM binding tests for VPCI indicator.
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
    // Load WASM module
    try {
        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);
        // No need to call default() for ES modules
    } catch (error) {
        console.error('Failed to load WASM module. Run "wasm-pack build --features wasm --target nodejs" first');
        throw error;
    }
    
    testData = loadTestData();
});

test('VPCI accuracy', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    // Using default parameters: short_range=5, long_range=25
    const result = wasm.vpci_js(close, volume, 5, 25);
    
    // Result is an object with vpci and vpcis arrays
    assert(result.vpci, 'Result should have vpci array');
    assert(result.vpcis, 'Result should have vpcis array');
    assert.strictEqual(result.vpci.length, close.length, 'VPCI length should match input length');
    assert.strictEqual(result.vpcis.length, close.length, 'VPCIS length should match input length');
    
    const vpci = result.vpci;
    const vpcis = result.vpcis;
    
    // Check last 5 values match expected from Rust tests
    const expectedLastFiveVpci = [
        -319.65148214323426,
        -133.61700649928346,
        -144.76194155503174,
        -83.55576212490328,
        -169.53504207700533,
    ];
    
    const expectedLastFiveVpcis = [
        -1049.2826640115732,
        -694.1067814399748,
        -519.6960416662324,
        -330.9401404636258,
        -173.004986803695,
    ];
    
    const actualLastFiveVpci = vpci.slice(-5);
    const actualLastFiveVpcis = vpcis.slice(-5);
    
    assertArrayClose(actualLastFiveVpci, expectedLastFiveVpci, 1e-1, 'VPCI last 5 values mismatch');
    assertArrayClose(actualLastFiveVpcis, expectedLastFiveVpcis, 1e-1, 'VPCIS last 5 values mismatch');
});

test('VPCI with default parameters', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    // Test with default values (short=5, long=25)
    const result = wasm.vpci_js(close, volume, 5, 25);
    assert(result.vpci, 'Result should have vpci array');
    assert(result.vpcis, 'Result should have vpcis array');
    assert.strictEqual(result.vpci.length, close.length, 'VPCI should match input length');
    assert.strictEqual(result.vpcis.length, close.length, 'VPCIS should match input length');
});

test('VPCI error handling - zero period', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    assert.throws(
        () => wasm.vpci_js(close, volume, 0, 25),
        /Invalid/,
        'Should throw error for zero short range'
    );
});

test('VPCI error handling - invalid period', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    assert.throws(
        () => wasm.vpci_js(close, volume, 25, 5),
        /Invalid/,
        'Should throw error when short_range > long_range'
    );
});

test('VPCI error handling - mismatched input lengths', () => {
    const close = testData.close;
    const volume = testData.volume.slice(0, -10); // Shorter volume array
    
    assert.throws(
        () => wasm.vpci_js(close, volume, 5, 25),
        /Mismatched input lengths/,
        'Should throw error for mismatched input lengths'
    );
});

test('VPCI error handling - insufficient data', () => {
    const singlePoint = new Float64Array([42.0]);
    
    assert.throws(
        () => wasm.vpci_js(singlePoint, singlePoint, 5, 25),
        /Invalid range/,
        'Should throw error for insufficient data'
    );
});

test('VPCI error handling - empty input', () => {
    const empty = new Float64Array([]);
    
    assert.throws(
        () => wasm.vpci_js(empty, empty, 5, 25),
        /All close or volume values are NaN/,
        'Should throw error for empty input'
    );
});

test('VPCI error handling - all NaN input', () => {
    const allNaN = new Float64Array(100).fill(NaN);
    
    assert.throws(
        () => wasm.vpci_js(allNaN, allNaN, 5, 25),
        /All close or volume values are NaN/,
        'Should throw error for all NaN values'
    );
});

test('VPCI NaN handling', () => {
    const close = testData.close;
    const volume = testData.volume;
    
    const result = wasm.vpci_js(close, volume, 5, 25);
    const vpci = result.vpci;
    const vpcis = result.vpcis;
    const len = close.length;
    
    // Check that we have expected number of NaN values at the beginning
    let nanCountVpci = 0;
    let nanCountVpcis = 0;
    
    for (let i = 0; i < len; i++) {
        if (isNaN(vpci[i])) {
            nanCountVpci++;
        } else {
            break;
        }
    }
    
    for (let i = 0; i < len; i++) {
        if (isNaN(vpcis[i])) {
            nanCountVpcis++;
        } else {
            break;
        }
    }
    
    // VPCI has a warmup period based on the longest period
    assert(nanCountVpci > 0, 'VPCI should have NaN values during warmup period');
    assert(nanCountVpcis > 0, 'VPCIS should have NaN values during warmup period');
    assert.strictEqual(nanCountVpci, nanCountVpcis, 'VPCI and VPCIS should have same warmup period');
});

test('VPCI batch processing', () => {
    const close = testData.close.slice(0, 100); // Use smaller dataset for speed
    const volume = testData.volume.slice(0, 100);
    
    // Test batch with single parameter set
    const config = {
        short_range: [5, 5, 0],
        long_range: [25, 25, 0]
    };
    
    const result = wasm.vpci_batch(close, volume, config);
    
    assert(result.vpci, 'Result should have vpci array');
    assert(result.vpcis, 'Result should have vpcis array');
    assert(result.combos, 'Result should have combos array');
    assert.strictEqual(result.rows, 1, 'Should have 1 row for single parameter set');
    assert.strictEqual(result.cols, 100, 'Should have 100 columns');
    
    // Compare with single calculation
    const singleResult = wasm.vpci_js(close, volume, 5, 25);
    const singleVpci = singleResult.vpci;
    const singleVpcis = singleResult.vpcis;
    
    assertArrayClose(
        result.vpci, 
        singleVpci, 
        1e-10, 
        'Batch VPCI should match single calculation'
    );
    assertArrayClose(
        result.vpcis, 
        singleVpcis, 
        1e-10, 
        'Batch VPCIS should match single calculation'
    );
});

test('VPCI batch with multiple parameters', () => {
    const close = testData.close.slice(0, 50); // Use smaller dataset for speed
    const volume = testData.volume.slice(0, 50);
    
    // Multiple parameter combinations
    const config = {
        short_range: [5, 10, 5],     // 5, 10
        long_range: [20, 30, 10]     // 20, 30
    };
    
    const result = wasm.vpci_batch(close, volume, config);
    
    // Should have 2 * 2 = 4 combinations
    assert.strictEqual(result.rows, 4, 'Should have 4 parameter combinations');
    assert.strictEqual(result.cols, 50, 'Should have 50 columns');
    assert.strictEqual(result.vpci.length, 200, 'VPCI should have 4 * 50 = 200 values');
    assert.strictEqual(result.vpcis.length, 200, 'VPCIS should have 4 * 50 = 200 values');
    
    // Check first combination matches single calculation
    const firstRowVpci = result.vpci.slice(0, 50);
    const firstRowVpcis = result.vpcis.slice(0, 50);
    const singleResult = wasm.vpci_js(close, volume, 5, 20);
    const singleVpci = singleResult.vpci;
    const singleVpcis = singleResult.vpcis;
    
    assertArrayClose(firstRowVpci, singleVpci, 1e-10, 'First batch VPCI row should match single calculation');
    assertArrayClose(firstRowVpcis, singleVpcis, 1e-10, 'First batch VPCIS row should match single calculation');
});

test('VPCI memory allocation/deallocation', () => {
    const len = 1000;
    const ptr = wasm.vpci_alloc(len);
    
    assert(ptr !== 0, 'Allocated pointer should not be null');
    
    // Free the memory
    wasm.vpci_free(ptr, len);
    
    // Test multiple allocations
    const ptrs = [];
    for (let i = 0; i < 10; i++) {
        ptrs.push(wasm.vpci_alloc(100));
    }
    
    // Free all
    ptrs.forEach(p => wasm.vpci_free(p, 100));
});

test('VPCI fast API (vpci_into)', () => {
    const close = testData.close.slice(0, 100);
    const volume = testData.volume.slice(0, 100);
    const len = close.length;
    
    // Allocate buffers for input and output
    const closePtr = wasm.vpci_alloc(len);
    const volumePtr = wasm.vpci_alloc(len);
    const vpciPtr = wasm.vpci_alloc(len);
    const vpcisPtr = wasm.vpci_alloc(len);
    
    try {
        // Copy input data to WASM memory
        const closeView = new Float64Array(wasm.__wasm.memory.buffer, closePtr, len);
        const volumeView = new Float64Array(wasm.__wasm.memory.buffer, volumePtr, len);
        closeView.set(close);
        volumeView.set(volume);
        
        // Call fast API with pointers
        wasm.vpci_into(
            closePtr,
            volumePtr,
            vpciPtr,
            vpcisPtr,
            len,
            5,
            25
        );
        
        // Read results from WASM memory
        const memoryVpci = new Float64Array(wasm.__wasm.memory.buffer, vpciPtr, len);
        const memoryVpcis = new Float64Array(wasm.__wasm.memory.buffer, vpcisPtr, len);
        const resultVpci = Array.from(memoryVpci);
        const resultVpcis = Array.from(memoryVpcis);
        
        // Compare with safe API
        const expected = wasm.vpci_js(close, volume, 5, 25);
        const expectedVpci = expected.vpci;
        const expectedVpcis = expected.vpcis;
        
        assertArrayClose(resultVpci, expectedVpci, 1e-14, 'Fast API VPCI should match safe API');
        assertArrayClose(resultVpcis, expectedVpcis, 1e-14, 'Fast API VPCIS should match safe API');
        
    } finally {
        wasm.vpci_free(closePtr, len);
        wasm.vpci_free(volumePtr, len);
        wasm.vpci_free(vpciPtr, len);
        wasm.vpci_free(vpcisPtr, len);
    }
});

test.after(() => {
    console.log('VPCI WASM tests completed');
});

export { };