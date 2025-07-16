import test from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData, assertArrayClose } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('reproduce exact ADXR batch test', async () => {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    const testData = loadTestData();
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Multiple periods: 10, 15, 20
    const batchResult = wasm.adxr_batch_js(
        high, low, close,
        10, 20, 5      // period range
    );
    
    // Should have 3 rows * 100 cols = 300 values
    assert.strictEqual(batchResult.length, 3 * 100);
    
    // Verify each row matches individual calculation
    const periods = [10, 15, 20];
    for (let i = 0; i < periods.length; i++) {
        const rowStart = i * 100;
        const rowEnd = rowStart + 100;
        const rowData = batchResult.slice(rowStart, rowEnd);
        
        const singleResult = wasm.adxr_js(high, low, close, periods[i]);
        
        console.log(`\nPeriod ${periods[i]}:`);
        console.log(`Row data at index 30: ${rowData[30]}`);
        console.log(`Single result at index 30: ${singleResult[30]}`);
        
        try {
            assertArrayClose(
                rowData, 
                singleResult, 
                1e-10, 
                `Period ${periods[i]} mismatch`
            );
            console.log(`Period ${periods[i]} passed`);
        } catch (e) {
            console.log(`Period ${periods[i]} FAILED:`, e.message);
        }
    }
});