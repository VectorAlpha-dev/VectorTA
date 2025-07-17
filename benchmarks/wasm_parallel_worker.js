// Worker script for parallel ALMA computation
const { parentPort, workerData } = require('worker_threads');

// Load WASM module in worker
function loadWasmInWorker() {
    const wasm = require('../pkg/my_project.js');
    return wasm;
}

function processAlmaTask(wasm, task) {
    try {
        const { data, length, period, offset, sigma } = task;
        
        if (!data || !length) {
            throw new Error('Invalid data: empty or undefined');
        }
        
        // Convert array back to Float64Array
        const typedData = new Float64Array(data);
        
        // Use pre-allocated buffer approach for best performance
        const ptr = wasm.alma_alloc(length);
        if (!ptr) {
            throw new Error('Failed to allocate WASM memory');
        }
        
        // Create view into WASM memory
        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            length
        );
        
        // Copy data into WASM memory
        memView.set(typedData);
        
        // Compute ALMA - alma_into expects both input and output pointers
        try {
            wasm.alma_into(ptr, ptr, length, period, offset, sigma);
            
            // Copy result back
            const output = new Float64Array(length);
            output.set(memView);
            
            // Free memory
            wasm.alma_free(ptr, length);
            
            // Convert back to regular array for transfer
            return Array.from(output);
        } catch (wasmError) {
            // Free memory on error
            wasm.alma_free(ptr, length);
            throw new Error(`ALMA computation failed: ${wasmError}`);
        }
    } catch (error) {
        throw new Error(`processAlmaTask failed: ${error.message || error}`);
    }
}

// Worker message handler
try {
    const wasm = loadWasmInWorker();
    
    parentPort.on('message', (message) => {
        const { type, taskId, payload } = message;
        
        if (type === 'compute') {
            try {
                const result = processAlmaTask(wasm, payload);
                parentPort.postMessage({
                    type: 'result',
                    taskId,
                    result
                });
            } catch (error) {
                parentPort.postMessage({
                    type: 'error',
                    taskId,
                    error: error.message || error.toString() || 'Unknown error'
                });
            }
        }
    });
    
    // Signal worker is ready
    parentPort.postMessage({ type: 'ready' });
    
} catch (error) {
    parentPort.postMessage({
        type: 'init_error',
        error: error.message
    });
}