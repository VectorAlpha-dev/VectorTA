import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Fix for Windows ESM import paths
const fixImportPath = `        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        wasm = await import(wasmPath);`;

const fixedImportPath = `        const wasmPath = path.join(__dirname, '../../pkg/my_project.js');
        const importPath = process.platform === 'win32' 
            ? 'file:///' + wasmPath.replace(/\\\\/g, '/')
            : wasmPath;
        wasm = await import(importPath);`;

async function fixTestFiles() {
    const testFiles = await fs.readdir(__dirname);
    const jsFiles = testFiles.filter(f => f.startsWith('test_') && f.endsWith('.js'));
    
    for (const file of jsFiles) {
        const filePath = path.join(__dirname, file);
        let content = await fs.readFile(filePath, 'utf8');
        
        if (content.includes(fixImportPath)) {
            content = content.replace(fixImportPath, fixedImportPath);
            await fs.writeFile(filePath, content);
            console.log(`Fixed: ${file}`);
        }
    }
}

fixTestFiles().catch(console.error);