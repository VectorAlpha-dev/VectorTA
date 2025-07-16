import test from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test('debug ADXR single vs batch', async () => {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    // Use actual test data
    const testData = loadTestData();
    const len = 100;
    const high = new Float64Array(testData.high.slice(0, len));
    const low = new Float64Array(testData.low.slice(0, len));
    const close = new Float64Array(testData.close.slice(0, len));
    
    const period = 15;
    
    // Single call
    const singleResult = wasm.adxr_js(high, low, close, period);
    
    // Batch call with just one period
    const batchResult = wasm.adxr_batch_js(high, low, close, period, period, 0);
    
    console.log('Period:', period);
    console.log('First ADX should be at:', 2 * period);
    console.log('First ADXR should be at:', 3 * period - 1, '(needs ADX from period bars ago)');
    
    // Find first non-zero value
    let firstNonZero = -1;
    for (let i = 0; i < len; i++) {
        if (!isNaN(singleResult[i]) && singleResult[i] !== 0) {
            firstNonZero = i;
            break;
        }
    }
    console.log('First non-zero value at index:', firstNonZero);
    
    console.log('\nSingle values [40-50]:', Array.from(singleResult.slice(40, 51)).map((v, i) => `[${40+i}]=${v.toFixed(2)}`).join(', '));
    console.log('Batch values [40-50]:', Array.from(batchResult.slice(40, 51)).map((v, i) => `[${40+i}]=${v.toFixed(2)}`).join(', '));
    
    // Check if they match
    for (let i = 0; i < len; i++) {
        const s = singleResult[i];
        const b = batchResult[i];
        if (isNaN(s) && isNaN(b)) continue;
        if (Math.abs(s - b) > 1e-10) {
            console.log(`Mismatch at index ${i}: single=${s}, batch=${b}`);
        }
    }
});