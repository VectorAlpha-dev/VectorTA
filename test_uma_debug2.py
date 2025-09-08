import numpy as np

# Pine formula test
for price in [100, 200, 1000, 10000, 59000]:
    for length in [5, 10, 20, 50]:
        mf = 100 - 100 / (1 + price / length)
        print(f"Price={price:5}, len={length:2}: mf={mf:.2f}")
    print()

print("When price >> length, mf approaches 100")
print("When price ≈ length, mf ≈ 50")
print("When price < length, mf < 50")
print()

# For typical crypto prices around 59000 with length around 5-50
# mf will be very close to 100
# This means p = acc + |mf*2-100|/25 = 1 + |100*2-100|/25 = 1 + 100/25 = 5

print("With price=59000, len=50: p = 1 + |99.92*2-100|/25 =", 1 + abs(99.92*2-100)/25)