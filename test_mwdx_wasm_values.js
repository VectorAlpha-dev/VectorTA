import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM module
const wasmPath = process.platform === 'win32' 
    ? 'file:///' + join(__dirname, 'pkg/my_project.js').replace(/\\/g, '/')
    : join(__dirname, 'pkg/my_project.js');
const wasm = await import(wasmPath);

// Test data
const testData = [10.0, 20.0, 30.0, 40.0, 50.0];
const factor = 0.2;

console.log("Testing MWDX WASM computation...");
console.log("Input data:", testData);
console.log("Factor:", factor);

// Test Safe API
console.log("\n--- Safe API ---");
const safeResult = wasm.mwdx_js(new Float64Array(testData), factor);
console.log("Result:", Array.from(safeResult));

// Test Fast API
console.log("\n--- Fast API ---");
const len = testData.length;
const inPtr = wasm.mwdx_alloc(len);
const outPtr = wasm.mwdx_alloc(len);

try {
    // For wasm-bindgen generated modules, memory might be on __wasm
    const memory = wasm.__wasm?.memory || wasm.memory || wasm.default?.memory;
    if (!memory) {
        console.log("ERROR: WASM memory not accessible - Fast API test skipped");
        console.log("Available exports:", Object.keys(wasm));
    } else {
        // Copy data to WASM memory
        new Float64Array(memory.buffer, inPtr, len).set(testData);
        
        // Compute
        wasm.mwdx_into(inPtr, outPtr, len, factor);
        
        // Read result
        const fastResult = new Float64Array(memory.buffer, outPtr, len);
        console.log("Result:", Array.from(fastResult));
    
    // Verify computation manually
    console.log("\n--- Manual Verification ---");
    const expected = [testData[0]];
    for (let i = 1; i < len; i++) {
        expected[i] = factor * testData[i] + (1 - factor) * expected[i - 1];
    }
    console.log("Expected:", expected);
    
    // Compare results
    let allMatch = true;
    for (let i = 0; i < len; i++) {
        if (Math.abs(fastResult[i] - expected[i]) > 1e-10) {
            console.log(`Mismatch at index ${i}: ${fastResult[i]} vs ${expected[i]}`);
            allMatch = false;
        }
    }
    console.log("\nResults match:", allMatch);
    }
    
} finally {
    wasm.mwdx_free(inPtr, len);
    wasm.mwdx_free(outPtr, len);
}