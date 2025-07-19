import { performance } from 'perf_hooks';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load WASM module
const wasmPath = process.platform === 'win32' 
    ? 'file:///' + join(__dirname, 'pkg/my_project.js').replace(/\\/g, '/')
    : join(__dirname, 'pkg/my_project.js');
const wasm = await import(wasmPath);

// Generate test data
const sizes = [10_000, 100_000, 1_000_000];
const factor = 0.2;
const iterations = 100;

console.log("Simple MWDX Benchmark");
console.log("====================");
console.log(`Iterations per test: ${iterations}`);
console.log(`Factor: ${factor}\n`);

for (const size of sizes) {
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.random() * 100;
    }
    
    console.log(`\nData size: ${size.toLocaleString()}`);
    console.log("-".repeat(40));
    
    // Benchmark Safe API
    const safeStart = performance.now();
    for (let i = 0; i < iterations; i++) {
        const result = wasm.mwdx_js(data, factor);
    }
    const safeTime = (performance.now() - safeStart) / iterations;
    console.log(`Safe API:  ${safeTime.toFixed(3)} ms per iteration`);
    
    // Benchmark Fast API
    const len = data.length;
    const inPtr = wasm.mwdx_alloc(len);
    const outPtr = wasm.mwdx_alloc(len);
    
    try {
        // Copy data once
        const memory = wasm.__wasm?.memory;
        if (memory) {
            new Float64Array(memory.buffer, inPtr, len).set(data);
            
            const fastStart = performance.now();
            for (let i = 0; i < iterations; i++) {
                wasm.mwdx_into(inPtr, outPtr, len, factor);
            }
            const fastTime = (performance.now() - fastStart) / iterations;
            console.log(`Fast API:  ${fastTime.toFixed(3)} ms per iteration`);
            console.log(`Speedup:   ${(safeTime / fastTime).toFixed(2)}x`);
        } else {
            console.log("Fast API:  Memory not accessible");
        }
    } finally {
        wasm.mwdx_free(inPtr, len);
        wasm.mwdx_free(outPtr, len);
    }
}