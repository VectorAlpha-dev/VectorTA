// Debug TEMA issue
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

console.log('Testing TEMA edge cases...');

// Test 1: Period equals data length (failing test from line 212)
console.log('\nTest 1: Period 10, data length 10');
try {
    const data = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result = wasm.tema_js(data, 10);
    console.log('Success! Result length:', result.length);
    console.log('First few values:', Array.from(result.slice(0, 5)));
} catch (error) {
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
}

// Test 2: Period 1 (failing test from line 220)
console.log('\nTest 2: Period 1');
try {
    const data2 = new Float64Array([1, 2, 3]);
    const result2 = wasm.tema_js(data2, 1);
    console.log('Success! Result:', Array.from(result2));
} catch (error) {
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
}

// Test 3: Very small dataset with period 1 (failing test from line 369)
console.log('\nTest 3: Period 1 with data length 1');
try {
    const data3 = new Float64Array([1]);
    const result3 = wasm.tema_js(data3, 1);
    console.log('Success! Result:', Array.from(result3));
} catch (error) {
    console.error('Error:', error.message);
    console.error('Stack:', error.stack);
}

// Test 4: Normal case that should work
console.log('\nTest 4: Normal case (should work)');
try {
    const data4 = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const result4 = wasm.tema_js(data4, 3);
    console.log('Success! Result:', Array.from(result4));
} catch (error) {
    console.error('Error:', error.message);
}