import { readFile } from 'fs/promises';

async function listOtherIndicators() {
  const content = await readFile('./src/data/indicator-registry.ts', 'utf-8');
  const match = content.match(/export const indicators = ({[\s\S]*});/);
  
  if (match) {
    const indicatorsObj = eval(`(${match[1]})`);
    
    console.log('Indicators in "other" category:');
    Object.entries(indicatorsObj)
      .filter(([, indicator]) => indicator.category === 'other')
      .forEach(([id, indicator]) => {
        console.log(`- ${id}: ${indicator.name}`);
      });
  }
}

listOtherIndicators().catch(console.error);