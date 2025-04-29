#!/usr/bin/env python3
"""
convert_to_fp64.py   infile.csv   outfile.csv

Re-writes the numeric part of a CSV so that every value after the header row
is rendered in full-precision float-64 form (no scientific E-notation unless
needed).  The header row is copied verbatim.
"""

import csv
import sys
from pathlib import Path

def main(in_path: Path, out_path: Path) -> None:
    # --- autodetect delimiter -------------------------------------------------
    with in_path.open("r", newline="") as f:
        sample = f.read(4096)
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)
    dialect.skipinitialspace = True

    # --- rewrite file ---------------------------------------------------------
    with in_path.open("r", newline="") as src, out_path.open("w", newline="") as dst:
        rdr = csv.reader(src, dialect)
        wtr = csv.writer(dst, dialect)

        # copy header untouched
        header = next(rdr)
        wtr.writerow(header)

        for row in rdr:
            fp_row = []
            for cell in row:
                cell = cell.strip()
                if cell == "":
                    fp_row.append("")              # allow blanks
                else:
                    try:
                        fp_row.append(f"{float(cell):.17g}")  # full-precision fp64
                    except ValueError:
                        fp_row.append(cell)        # non-numeric (shouldn’t happen)
            wtr.writerow(fp_row)

    print(f"✔ wrote {out_path} with all data cells as FP-64")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: convert_to_fp64.py  infile.csv  outfile.csv")
    main(Path(sys.argv[1]), Path(sys.argv[2]))
