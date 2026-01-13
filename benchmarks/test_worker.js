
const { Worker } = require('worker_threads');
const path = require('path');

console.log('Testing worker loading...');

const worker = new Worker(path.join(__dirname, 'wasm_parallel_worker.js'));

worker.on('message', (msg) => {
    console.log('Worker message:', msg);
    if (msg.type === 'ready') {
        console.log('Worker is ready!');


        const testData = new Float64Array([1, 2, 3, 4, 5]);
        worker.postMessage({
            type: 'compute',
            taskId: 1,
            payload: {
                data: Array.from(testData),
                length: testData.length,
                period: 3,
                offset: 0.85,
                sigma: 6.0
            }
        });
    }
});

worker.on('error', (err) => {
    console.error('Worker error:', err);
});

worker.on('exit', (code) => {
    console.log('Worker exited with code:', code);
});


setTimeout(() => {
    console.log('Test timeout - terminating worker');
    worker.terminate();
}, 5000);