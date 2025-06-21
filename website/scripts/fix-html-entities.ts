import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

async function fixHtmlEntities() {
  const indicatorsDir = './src/pages/indicators';
  const files = await readdir(indicatorsDir);
  
  for (const file of files) {
    if (file.endsWith('.astro')) {
      const filePath = join(indicatorsDir, file);
      let content = await readFile(filePath, 'utf-8');
      
      // Fix unescaped < and > characters
      // Don't replace those that are part of HTML tags or already escaped
      content = content.replace(/([^&])(<)(\s*[-\d])/g, '$1&lt;$3');
      content = content.replace(/([^&])(>)(\s*[-\d])/g, '$1&gt;$3');
      
      // Also fix standalone < and > in text
      content = content.replace(/(\s)(<)(\s)/g, '$1&lt;$3');
      content = content.replace(/(\s)(>)(\s)/g, '$1&gt;$3');
      
      await writeFile(filePath, content);
      console.log(`Fixed ${file}`);
    }
  }
}

fixHtmlEntities().catch(console.error);