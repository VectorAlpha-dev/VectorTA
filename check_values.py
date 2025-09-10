import csv

# Load the CSV data
rows = []
with open('src/data/2018-09-01-2024-Bitfinex_Spot-4h.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)

print(f"Total rows: {len(rows)}")

# Find where close prices are around 68k
high_price_indices = []
for i, row in enumerate(rows):
    close_price = float(row[2])  # close is column 2
    if 65000 < close_price < 70000:
        high_price_indices.append(i)

if high_price_indices:
    print(f"\nFound {len(high_price_indices)} rows with close between 65k-70k")
    print(f"First at index: {high_price_indices[0]}")
    print(f"Last at index: {high_price_indices[-1]}")
    
    # Show the last few
    print(f"\nLast 5 indices with 65-70k prices:")
    for idx in high_price_indices[-5:]:
        row = rows[idx]
        print(f"  [{idx}]: close={float(row[2]):.2f}, high={float(row[3]):.2f}, low={float(row[4]):.2f}")
else:
    print("\nNo rows found with close price between 65k-70k")

# Show the last 5 rows of the dataset
print("\nLast 5 rows of dataset:")
for i in range(len(rows)-5, len(rows)):
    row = rows[i]
    print(f"  [{i}]: close={float(row[2]):.2f}, high={float(row[3]):.2f}, low={float(row[4]):.2f}")