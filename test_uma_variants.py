import numpy as np

# Test different interpretations of the fallback formula

prices = [59000, 59100, 59200, 59300, 59400]
length = 20

print("Different interpretations of Pine fallback formula:\n")

# 1. As given: mf = 100 - 100/(1 + src/len)
print("1. Original interpretation (src=price, len=period):")
for p in prices:
    mf = 100 - 100 / (1 + p / length)
    print(f"  Price {p}: mf = {mf:.2f}")

# 2. Maybe it's price change over period?
print("\n2. If src = price_change_over_period:")
price_changes = [0.5, 1.0, 2.0, -0.5, -1.0]  # percentages
for pc in price_changes:
    mf = 100 - 100 / (1 + abs(pc) / length)
    print(f"  Change {pc:+.1f}%: mf = {mf:.2f}")

# 3. Maybe it's meant to be RSI-like: 100 / (1 + RS) where RS = avg_gain/avg_loss
print("\n3. RSI-like formula (100 / (1 + losses/gains)):")
for ratio in [0.5, 1.0, 2.0, 3.0]:
    mf = 100 / (1 + ratio)
    print(f"  Loss/Gain ratio {ratio}: mf = {mf:.2f}")

# 4. Maybe the formula should be simpler momentum
print("\n4. Simple momentum (price / price_n_bars_ago * 100 - 100):")
for i in range(1, len(prices)):
    mom = (prices[i] / prices[0]) * 100 - 100
    # Convert to 0-100 scale
    mf = 50 + mom * 10  # scale factor
    mf = max(0, min(100, mf))  # clamp
    print(f"  Price {prices[i]} vs {prices[0]}: momentum = {mom:.2f}%, mf = {mf:.2f}")