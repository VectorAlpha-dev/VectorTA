import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load WASM module
const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
const wasm = await import(wasmPath);

// Test data
const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

console.log('Testing hma_js with period 3:');
try {
    const result = wasm.hma_js(data, 3);
    console.log('Success:', result.length, 'values');
} catch (e) {
    console.error('Error:', e.message);
}

console.log('\nTesting hma_batch_js with periods 3-5:');
try {
    const result = wasm.hma_batch_js(data, 3, 5, 2);
    console.log('Success:', result.length, 'values');
} catch (e) {
    console.error('Error:', e.message);
}

console.log('\nTesting hma_batch_metadata_js:');
try {
    const result = wasm.hma_batch_metadata_js(3, 5, 2);
    console.log('Success:', result);
} catch (e) {
    console.error('Error:', e.message);
}
