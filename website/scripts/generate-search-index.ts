import { readdir, readFile, writeFile, stat } from 'fs/promises';
import { join } from 'path';
import matter from 'gray-matter';
import type { SearchItem } from '../src/components/search/search-types';

async function generateSearchIndex() {
  const searchItems: SearchItem[] = [];
  const indicatorsDir = 'src/content/indicators';
  
  // Recursive function to find all MDX files
  async function findMDXFiles(dir: string): Promise<string[]> {
    const files: string[] = [];
    const entries = await readdir(dir);
    
    for (const entry of entries) {
      const fullPath = join(dir, entry);
      const entryStat = await stat(fullPath);
      
      if (entryStat.isDirectory()) {
        // Recursively search subdirectories
        const subFiles = await findMDXFiles(fullPath);
        files.push(...subFiles);
      } else if (entry.endsWith('.mdx')) {
        files.push(fullPath);
      }
    }
    
    return files;
  }
  
  // Find all MDX files
  const mdxFiles = await findMDXFiles(indicatorsDir);
  
  for (const filePath of mdxFiles) {
    const content = await readFile(filePath, 'utf-8');
    const { data } = matter(content);
    
    // Extract ID from file path
    const relativePath = filePath.replace('src/content/indicators/', '');
    const id = relativePath.replace('.mdx', '').split('/').pop() || '';
    
    searchItems.push({
      id,
      title: data.title || id,
      description: data.description || '',
      category: data.category || 'other',
      url: `/indicators/${id}`,
      type: 'indicator'
    });
  }
  
  // Add static pages
  searchItems.push(
    {
      id: 'home',
      title: 'Home',
      description: 'VectorTA - High-performance technical analysis library',
      category: 'page',
      url: '/',
      type: 'page'
    },
    {
      id: 'indicators',
      title: 'All Indicators',
      description: 'Browse all technical analysis indicators',
      category: 'page',
      url: '/indicators',
      type: 'page'
    }
  );
  
  // Write search index
  const output = `// Auto-generated search index
export const searchIndex: SearchItem[] = ${JSON.stringify(searchItems, null, 2)};

export interface SearchItem {
  id: string;
  title: string;
  description: string;
  category: string;
  url: string;
  type: 'indicator' | 'guide' | 'page';
}
`;
  
  await writeFile('src/data/search-index.ts', output);
  console.log(`Generated search index with ${searchItems.length} items`);
}

generateSearchIndex().catch(console.error);