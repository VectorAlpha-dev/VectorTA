
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


const wasmPath = path.join(__dirname, '../../pkg/vector_ta.js');
const importPath = process.platform === 'win32' 
    ? 'file:///' + wasmPath.replace(/\\/g, '/')
    : wasmPath;
const wasm = await import(importPath);


console.log('Testing TEMA with period=2, data=[1,2]');
try {
    const data = new Float64Array([1, 2]);
    console.log('Calling tema_js...');
    const result = wasm.tema_js(data, 2);
    console.log('Success! Result:', Array.from(result));
} catch (error) {
    console.error('Error:', error.message);
    console.error('Full error:', error);
    
    
    if (error.stack) {
        const lines = error.stack.split('\n');
        console.log('\nStack trace:');
        lines.forEach(line => console.log('  ', line));
    }
}


console.log('\n\nTesting with zero-copy API...');
try {
    const data = new Float64Array([1, 2]);
    const ptr = wasm.tema_alloc(2);
    const memView = new Float64Array(wasm.__wasm.memory.buffer, ptr, 2);
    memView.set(data);
    
    console.log('Calling tema_into...');
    wasm.tema_into(ptr, ptr, 2, 2);
    console.log('Success! Result:', Array.from(memView));
    
    wasm.tema_free(ptr, 2);
} catch (error) {
    console.error('Error:', error.message);
}