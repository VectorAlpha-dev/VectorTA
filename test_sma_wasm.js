import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function test() {
    // Load WASM module
    const wasmPath = join(__dirname, 'pkg/my_project.js');
    const wasm = await import(`file:///${wasmPath.replace(/\\/g, '/')}`);
    
    // Create test data
    const testData = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const period = 3;
    
    console.log('Test data:', Array.from(testData));
    
    // Test safe API
    console.log('\n--- Safe API Test ---');
    const safeResult = wasm.sma(testData, period);
    console.log('Safe API result:', Array.from(safeResult));
    
    // Test fast API
    console.log('\n--- Fast API Test ---');
    const len = testData.length;
    const inPtr = wasm.sma_alloc(len);
    const outPtr = wasm.sma_alloc(len);
    
    try {
        // Copy data to WASM memory
        const inView = new Float64Array(wasm.__wasm.memory.buffer, inPtr, len);
        inView.set(testData);
        
        // Call fast API
        wasm.sma_into(inPtr, outPtr, len, period);
        
        // Read result
        const outView = new Float64Array(wasm.__wasm.memory.buffer, outPtr, len);
        const fastResult = Array.from(outView);
        console.log('Fast API result:', fastResult);
        
        // Compare results
        console.log('\n--- Comparison ---');
        const match = safeResult.every((val, i) => 
            (isNaN(val) && isNaN(fastResult[i])) || Math.abs(val - fastResult[i]) < 1e-10
        );
        console.log('Results match:', match);
        
        // Expected values for SMA with period 3:
        // [NaN, NaN, 2, 3, 4, 5, 6, 7, 8, 9]
        console.log('\nExpected: [NaN, NaN, 2, 3, 4, 5, 6, 7, 8, 9]');
        
    } finally {
        wasm.sma_free(inPtr, len);
        wasm.sma_free(outPtr, len);
    }
}

test().catch(console.error);