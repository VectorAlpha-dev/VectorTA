import { readFile } from 'fs/promises';

async function countCategories() {
  const content = await readFile('./src/data/indicator-registry.ts', 'utf-8');
  const match = content.match(/export const indicators = ({[\s\S]*});/);
  
  if (match) {
    const indicatorsObj = eval(`(${match[1]})`);
    const categoryCounts = {};
    
    Object.values(indicatorsObj).forEach(indicator => {
      const category = indicator.category || 'other';
      categoryCounts[category] = (categoryCounts[category] || 0) + 1;
    });
    
    console.log('Category counts:');
    Object.entries(categoryCounts)
      .sort(([,a], [,b]) => b - a)
      .forEach(([category, count]) => {
        console.log(`${category}: ${count}`);
      });
    
    console.log(`\nTotal indicators: ${Object.keys(indicatorsObj).length}`);
  }
}

countCategories().catch(console.error);