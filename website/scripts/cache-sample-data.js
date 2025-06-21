import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { dirname } from 'path';
import Papa from 'papaparse';

async function cacheSampleData() {
  try {
    // Ensure public/data directory exists
    const dataDir = './public/data';
    if (!existsSync(dataDir)) {
      await mkdir(dataDir, { recursive: true });
    }
    
    // Try to load a CSV file
    let csvContent;
    const possiblePaths = [
      './2018-09-01-2024-Bitfinex_Spot-4h.csv',
      '../src/data/10kCandles.csv',
      '../../src/data/10kCandles.csv'
    ];
    
    for (const path of possiblePaths) {
      try {
        csvContent = await readFile(path, 'utf-8');
        console.log(`Found CSV file at: ${path}`);
        break;
      } catch (e) {
        // Continue to next path
      }
    }
    
    if (!csvContent) {
      // Generate mock data if no CSV found
      console.log('No CSV file found, generating mock data...');
      const mockData = generateMockData();
      await writeFile('./public/data/sample-ohlcv.json', JSON.stringify(mockData));
      console.log(`Generated ${mockData.length} mock candles for demo`);
      return;
    }
    
    // Parse CSV
    const parsed = Papa.parse(csvContent, {
      header: false,
      dynamicTyping: true,
      skipEmptyLines: true
    });
    
    // Skip header row if it exists
    const dataRows = parsed.data.filter(row => 
      row.length >= 6 && !isNaN(Number(row[0]))
    );
    
    // Convert to OHLCV format
    const ohlcvData = dataRows.slice(-1000).map(row => ({
      time: Math.floor(Number(row[0]) / 1000), // Convert ms to seconds
      open: parseFloat(row[1]),
      high: parseFloat(row[2]),
      low: parseFloat(row[4]),
      close: parseFloat(row[3]),
      volume: parseFloat(row[5])
    }));
    
    // Write to public directory
    await writeFile('./public/data/sample-ohlcv.json', JSON.stringify(ohlcvData));
    console.log(`Cached ${ohlcvData.length} candles for demo`);
    
  } catch (error) {
    console.error('Error caching sample data:', error);
    // Generate fallback data
    const mockData = generateMockData();
    await writeFile('./public/data/sample-ohlcv.json', JSON.stringify(mockData));
    console.log('Generated fallback mock data');
  }
}

function generateMockData() {
  const data = [];
  let time = Math.floor(Date.now() / 1000) - 1000 * 14400; // Start 1000 4h candles ago
  let price = 30000 + Math.random() * 10000;
  
  for (let i = 0; i < 1000; i++) {
    const change = (Math.random() - 0.5) * price * 0.02;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * price * 0.005;
    const low = Math.min(open, close) - Math.random() * price * 0.005;
    const volume = 100 + Math.random() * 1000;
    
    data.push({
      time: time + i * 14400, // 4 hour candles
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
      volume: Math.round(volume * 100) / 100
    });
    
    price = close;
  }
  
  return data;
}

cacheSampleData().catch(console.error);