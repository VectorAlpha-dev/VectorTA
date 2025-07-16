import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function trace() {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    const testData = loadTestData();
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    // Test period 15
    const result = wasm.adxr_js(high, low, close, 15);
    
    console.log('ADXR with period 15:');
    console.log('First 50 values:');
    for (let i = 0; i < 50; i++) {
        if (!isNaN(result[i]) && result[i] !== 0) {
            console.log(`[${i}] = ${result[i].toFixed(4)}`);
        }
    }
    
    // Find where NaN stops
    let lastNaN = -1;
    for (let i = 0; i < 100; i++) {
        if (isNaN(result[i])) {
            lastNaN = i;
        }
    }
    console.log('\nLast NaN at index:', lastNaN);
    console.log('Expected last NaN at:', 15 * 2 - 1, '(2 * period - 1)');
}

trace().catch(console.error);