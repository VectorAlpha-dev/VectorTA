
const { parentPort, workerData } = require('worker_threads');


function loadWasmInWorker() {
    const wasm = require('../pkg/vector_ta.js');
    return wasm;
}

function processAlmaTask(wasm, task) {
    try {
        const { data, length, period, offset, sigma } = task;

        if (!data || !length) {
            throw new Error('Invalid data: empty or undefined');
        }


        const typedData = new Float64Array(data);


        const ptr = wasm.alma_alloc(length);
        if (!ptr) {
            throw new Error('Failed to allocate WASM memory');
        }


        const memView = new Float64Array(
            wasm.__wasm.memory.buffer,
            ptr,
            length
        );


        memView.set(typedData);


        try {
            wasm.alma_into(ptr, ptr, length, period, offset, sigma);


            const output = new Float64Array(length);
            output.set(memView);


            wasm.alma_free(ptr, length);


            return Array.from(output);
        } catch (wasmError) {

            wasm.alma_free(ptr, length);
            throw new Error(`ALMA computation failed: ${wasmError}`);
        }
    } catch (error) {
        throw new Error(`processAlmaTask failed: ${error.message || error}`);
    }
}


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


    parentPort.postMessage({ type: 'ready' });

} catch (error) {
    parentPort.postMessage({
        type: 'init_error',
        error: error.message
    });
}