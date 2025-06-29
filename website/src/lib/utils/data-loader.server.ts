import Papa from 'papaparse';
import { readFile } from 'fs/promises';

export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export async function loadSampleData(size: 'small' | 'medium' | 'large' = 'small'): Promise<OHLCV[]> {
  // Check for local CSV file first
  const localCsvPath = './2018-09-01-2024-Bitfinex_Spot-4h.csv';
  let csvContent: string;
  
  try {
    csvContent = await readFile(localCsvPath, 'utf-8');
  } catch {
    // Fallback to parent directory files
    const fileMap = {
      small: '../src/data/10kCandles.csv',
      medium: '../src/data/bitfinex btc-usd 100,000 candles ends 09-01-24.csv',
      large: '../src/data/1MillionCandles.csv'
    };
    csvContent = await readFile(fileMap[size], 'utf-8');
  }
  
  const parsed = Papa.parse(csvContent, {
    header: false,
    dynamicTyping: true,
    skipEmptyLines: true
  });
  
  // Skip header row if it exists
  const dataRows = (parsed.data as any[]).filter((row: any) => 
    Array.isArray(row) && row.length >= 6 && !isNaN(Number(row[0]))
  );
  
  return dataRows.map((row: any) => ({
    time: Math.floor(Number(row[0]) / 1000), // Convert ms to seconds
    open: parseFloat(row[1]),
    high: parseFloat(row[2]),
    close: parseFloat(row[3]),
    low: parseFloat(row[4]),
    volume: parseFloat(row[5])
  }));
}

// Cache sample data for client-side use
export async function cacheSampleData() {
  try {
    const data = await loadSampleData('small');
    // Take last 1000 candles for demo
    const demoData = data.slice(-1000);
    
    // Write to public directory for client access
    const { writeFileSync } = await import('fs');
    writeFileSync('./public/data/sample-ohlcv.json', JSON.stringify(demoData));
    console.log(`Cached ${demoData.length} candles for demo`);
  } catch (error) {
    console.error('Error caching sample data:', error);
  }
}