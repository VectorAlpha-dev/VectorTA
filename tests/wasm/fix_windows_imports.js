import { promises as fs } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function fixWindowsImports() {
    console.log('Fixing Windows import paths in WASM test files...\n');
    
    const files = await fs.readdir(__dirname);
    const testFiles = files.filter(f => f.startsWith('test_') && f.endsWith('.js'));
    
    let fixedCount = 0;
    let alreadyFixedCount = 0;
    let skippedCount = 0;
    
    for (const file of testFiles) {
        const filePath = path.join(__dirname, file);
        let content = await fs.readFile(filePath, 'utf8');
        
        // Check if already fixed
        if (content.includes('process.platform === \'win32\'')) {
            console.log(`✓ ${file} - already fixed`);
            alreadyFixedCount++;
            continue;
        }
        
        // Look for the import pattern with regex
        const importRegex = /(\s*const wasmPath = path\.join\(__dirname, '\.\.\/\.\.\/pkg\/my_project\.js'\);\s*\n)(\s*)(wasm = await import\(wasmPath\);)/;
        
        if (importRegex.test(content)) {
            content = content.replace(importRegex, (match, pathLine, indent, importLine) => {
                return `${pathLine}${indent}const importPath = process.platform === 'win32' \n${indent}    ? 'file:///' + wasmPath.replace(/\\\\/g, '/')\n${indent}    : wasmPath;\n${indent}wasm = await import(importPath);`;
            });
            
            await fs.writeFile(filePath, content, 'utf8');
            console.log(`✓ ${file} - fixed`);
            fixedCount++;
        } else {
            // Skip test_utils.js as it doesn't need this pattern
            if (file === 'test_utils.js') {
                skippedCount++;
            } else {
                console.log(`⚠ ${file} - pattern not found`);
            }
        }
    }
    
    console.log(`\nSummary:`);
    console.log(`- Fixed: ${fixedCount} files`);
    console.log(`- Already fixed: ${alreadyFixedCount} files`);
    console.log(`- Skipped: ${skippedCount} files`);
    console.log(`- Total test files: ${testFiles.length}`);
}

fixWindowsImports().catch(error => {
    console.error('Error:', error);
    process.exit(1);
});