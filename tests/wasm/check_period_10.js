import { fileURLToPath } from 'url';
import path from 'path';
import { loadTestData } from './test_utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function check() {
    const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
    const importPath = process.platform === 'win32' 
        ? 'file:///' + wasmPath.replace(/\\/g, '/')
        : wasmPath;
    const wasm = await import(importPath);
    
    const testData = loadTestData();
    const high = new Float64Array(testData.high.slice(0, 100));
    const low = new Float64Array(testData.low.slice(0, 100));
    const close = new Float64Array(testData.close.slice(0, 100));
    
    console.log('Testing period 10:');
    const result10 = wasm.adxr_js(high, low, close, 10);
    console.log('Value at index 30:', result10[30]);
    console.log('Values 28-32:', result10.slice(28, 33));
    
    console.log('\nTesting period 15:');
    const result15 = wasm.adxr_js(high, low, close, 15);
    console.log('Value at index 30:', result15[30]);
    console.log('Values 28-32:', result15.slice(28, 33));
}

check().catch(console.error);