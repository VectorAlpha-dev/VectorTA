import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function debug() {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    const testData = loadTestData();
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test batch with periods 10, 15, 20
    const batchResult = wasm.adxr_batch_js(high, low, close, 10, 20, 5);
    
    // Extract row for period 15 (index 1)
    const row15 = batchResult.slice(100, 200);
    
    // Compare with single call
    const singleResult = wasm.adxr_js(high, low, close, 15);
    
    console.log('Batch result for period 15:');
    console.log('Index 30:', row15[30]);
    console.log('Values around index 30:', row15.slice(28, 33));
    
    console.log('\nSingle result for period 15:');
    console.log('Index 30:', singleResult[30]);
    console.log('Values around index 30:', singleResult.slice(28, 33));
    
    // Check if the value looks like garbage
    if (Math.abs(row15[30] - 4359.2) < 0.1) {
        console.log('\nFound the garbage value 4359.2!');
        console.log('This suggests uninitialized memory in the batch calculation');
        
        // Check hex representation
        const buf = new ArrayBuffer(8);
        const view = new DataView(buf);
        view.setFloat64(0, row15[30], true);
        const hex = Array.from(new Uint8Array(buf))
            .map(b => b.toString(16).padStart(2, '0'))
            .join(' ');
        console.log('Hex representation:', hex);
    }
}

debug().catch(console.error);