import { readdir, readFile, writeFile } from 'fs/promises';
import { join } from 'path';

async function fixIndicatorLayout() {
  const indicatorsDir = './src/pages/indicators';
  const files = await readdir(indicatorsDir);
  
  for (const file of files) {
    if (file.endsWith('.astro')) {
      const filePath = join(indicatorsDir, file);
      let content = await readFile(filePath, 'utf-8');
      
      // Fix the IndicatorLayout tag that has an extra >
      content = content.replace(
        /parameters={parameters}\s*\n\s*&gt;/g,
        'parameters={parameters}\n>'
      );
      
      // Also fix any <h3> or <h2> tags that got incorrectly modified
      content = content.replace(/<h2&gt;/g, '<h2>');
      content = content.replace(/<h3&gt;/g, '<h3>');
      content = content.replace(/<p&gt;/g, '<p>');
      content = content.replace(/<br\/&gt;/g, '<br/>');
      
      // Fix any table cells that got incorrectly modified
      content = content.replace(/<td class="py-2"&gt;/g, '<td class="py-2">');
      
      // Fix any code block paragraphs
      content = content.replace(/<p class="font-mono text-sm mb-0 text-gray-800 dark:text-gray-200"&gt;/g, 
                               '<p class="font-mono text-sm mb-0 text-gray-800 dark:text-gray-200">');
      
      await writeFile(filePath, content);
      console.log(`Fixed ${file}`);
    }
  }
}

fixIndicatorLayout().catch(console.error);