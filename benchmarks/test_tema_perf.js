// Simple TEMA performance test
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load WASM module
const wasmPath = path.join(__dirname, '../pkg/my_project.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);

// Create test data - 1M points
const close1M = new Float64Array(1000000);
for (let i = 0; i < close1M.length; i++) {
    close1M[i] = 100 + Math.sin(i * 0.01) * 10 + Math.random() * 2;
}

console.log('Testing TEMA performance with', close1M.length, 'data points');

// Test safe API
console.log('\n--- Safe API ---');
let start = performance.now();
for (let i = 0; i < 10; i++) {
    const result = wasm.tema_js(close1M, 14);
}
let elapsed = (performance.now() - start) / 10;
console.log('Safe API: ' + elapsed.toFixed(3) + 'ms');

// Test fast API
console.log('\n--- Fast API ---');
const ptr = wasm.tema_alloc(close1M.length);
const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, close1M.length);
memView.set(close1M);

start = performance.now();
for (let i = 0; i < 10; i++) {
    wasm.tema_into(ptr, ptr, close1M.length, 14);
}
elapsed = (performance.now() - start) / 10;
console.log('Fast API: ' + elapsed.toFixed(3) + 'ms');

wasm.tema_free(ptr, close1M.length);

// Compare with ALMA (offset must be <= period/2)
console.log('\n--- Comparison with ALMA ---');
start = performance.now();
for (let i = 0; i < 10; i++) {
    const result = wasm.alma_js(close1M, 14, 0.85, 6);
}
elapsed = (performance.now() - start) / 10;
console.log('ALMA Safe API: ' + elapsed.toFixed(3) + 'ms');

const almaPtr = wasm.alma_alloc(close1M.length);
const almaView = new Float64Array(wasm.__wasm.memory.buffer, almaPtr, close1M.length);
almaView.set(close1M);

start = performance.now();
for (let i = 0; i < 10; i++) {
    wasm.alma_into(almaPtr, almaPtr, close1M.length, 14, 0.85, 6);
}
elapsed = (performance.now() - start) / 10;
console.log('ALMA Fast API: ' + elapsed.toFixed(3) + 'ms');

wasm.alma_free(almaPtr, close1M.length);