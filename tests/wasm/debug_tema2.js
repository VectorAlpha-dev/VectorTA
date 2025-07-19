// Debug TEMA issue with more detail
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load WASM module
const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);

console.log('Testing TEMA with various period/data length combinations...');

// Test different combinations of period and data length
const testCases = [
    { period: 1, dataLen: 1 },    // Works
    { period: 1, dataLen: 3 },    // Works
    { period: 1, dataLen: 10 },   // ?
    { period: 2, dataLen: 2 },    // ?
    { period: 3, dataLen: 3 },    // ?
    { period: 5, dataLen: 5 },    // ?
    { period: 9, dataLen: 10 },   // ?
    { period: 10, dataLen: 10 },  // Fails
    { period: 10, dataLen: 11 },  // ?
    { period: 10, dataLen: 100 }, // ?
];

for (const {period, dataLen} of testCases) {
    console.log(`\nTesting period=${period}, dataLen=${dataLen}`);
    try {
        const data = new Float64Array(dataLen);
        for (let i = 0; i < dataLen; i++) {
            data[i] = i + 1;
        }
        
        const result = wasm.tema_js(data, period);
        const warmup = (period - 1) * 3;
        const numNaN = result.filter(x => isNaN(x)).length;
        console.log(`  Success! Warmup=${warmup}, NaN count=${numNaN}/${dataLen}`);
        
        // Show first non-NaN value if any
        const firstNonNaN = result.findIndex(x => !isNaN(x));
        if (firstNonNaN >= 0) {
            console.log(`  First non-NaN at index ${firstNonNaN}: ${result[firstNonNaN]}`);
        }
    } catch (error) {
        console.error(`  ERROR: ${error.message}`);
        if (error.message === 'unreachable') {
            console.error('  Stack trace indicates WASM unreachable panic');
        }
    }
}