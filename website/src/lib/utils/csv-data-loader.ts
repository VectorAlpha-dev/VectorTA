// src/lib/utils/csv-data-loader.ts
import Papa from 'papaparse';

export interface CandlestickData {
  /** Unix timestamp in SECONDS – required by Lightweight-Charts */
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

const SEC_1971 = 31_536_000;          // skip dummy rows like "0,1,2,3,4,5"

/**
 * Parse a remote CSV into CandlestickData[] suitable for Lightweight-Charts.
 * The CSV must have: 0-timestamp(ms), 1-open, 2-close, 3-high, 4-low, 5-volume
 */
export async function loadCSVData(
  filePath: string,
): Promise<CandlestickData[]> {
  /* ─── 1. Fetch ──────────────────────────────────────────────────────────── */
  // Add base path if the file path doesn't start with http
  let fullPath = filePath;
  if (!filePath.startsWith('http') && !filePath.startsWith('/')) {
    // Always use the base path (Astro requires it even in dev mode)
    const base = import.meta.env.BASE_URL || '/';
    fullPath = base.endsWith('/') ? `${base}${filePath}` : `${base}/${filePath}`;
  }
  
  const res = await fetch(fullPath);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while fetching ${fullPath}`);
  }
  const csvText = await res.text();

  /* ─── 2. Parse & sanitise ───────────────────────────────────────────────── */
  return new Promise((resolve, reject) => {
    Papa.parse(csvText, {
      delimiter: ',',
      skipEmptyLines: true,               // ignores blank lines :contentReference[oaicite:0]{index=0}
      dynamicTyping: true,                // converts numeric-looking cells :contentReference[oaicite:1]{index=1}
      complete: ({ data }) => {
        const clean: CandlestickData[] = [];

        (data as any[][]).forEach((row, idx) => {
          if (row.length < 6) {
            console.warn(`Row ${idx} too short → skipped`, row);
            return;
          }

          const tsMs = row[0];            // already a number if dynamicTyping worked
          if (!Number.isFinite(tsMs)) {
            console.warn(`Row ${idx} bad timestamp → skipped`, row);
            return;
          }

          const tsSec = Math.trunc(tsMs / 1000);
          if (tsSec <= SEC_1971) {        // filters “0,1,2,3,4,5” header row :contentReference[oaicite:2]{index=2}
            console.warn(`Row ${idx} pre-1971 or dummy → skipped`, row);
            return;
          }

          const [ , open, close, high, low, volume ] = row.map(Number);
          if ([open, high, low, close].some(v => !Number.isFinite(v))) {
            console.warn(`Row ${idx} contains NaN price → skipped`, row);
            return;
          }

          clean.push({ time: tsSec, open, high, low, close, volume });
        });

        /* final guard: sort & de-dupe for Lightweight-Charts requirements */
        clean.sort((a, b) => a.time - b.time);                      // ascending :contentReference[oaicite:3]{index=3}
        for (let i = 1; i < clean.length; i++) {
          if (clean[i].time === clean[i - 1].time) {
            console.warn(
              `Duplicate timestamp ${clean[i].time} at row ${i} → dropped`,
            );
            clean.splice(i--, 1);
          }
        }

        if (!clean.length) return reject(new Error('No valid candle data')); // propagates to chart

        resolve(clean);
      },
      error: reject,
    });
  });
}

/* Optional helper – unchanged */
export const getDataSubset = (
  data: CandlestickData[],
  count = 1_000,
): CandlestickData[] => (data.length <= count ? data : data.slice(-count));
