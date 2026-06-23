import numpy as np

def print_diff(pw, pdraw, pl):
    # Proportional
    denom = pw + pl
    p_adv_prop = pw / denom if denom > 0 else 0.5
    
    # Coinflip (50/50 resolution of draws)
    # This assumes ET+Pens are roughly a coinflip
    p_adv_coin = pw + pdraw * 0.5
    
    # "Reality" estimation (50% of draws decided in ET proportionally, 50% go to pens)
    p_adv_real = pw + (pdraw * 0.5 * (pw/denom)) + (pdraw * 0.5 * 0.5)

    print(f"90m: Win={pw:.2f}, Draw={pdraw:.2f}, Loss={pl:.2f}")
    print(f"  Advancement (Proportional) : {p_adv_prop:.4f}")
    print(f"  Advancement (Coinflip)     : {p_adv_coin:.4f}")
    print(f"  Advancement (Mixed Reality): {p_adv_real:.4f}")
    print(f"  Difference (Prop - Coin)   : {p_adv_prop - p_adv_coin:+.4f}")
    print(f"  Difference (Prop - Real)   : {p_adv_prop - p_adv_real:+.4f}\n")

print("--- Heavy Favorite ---")
print_diff(0.70, 0.20, 0.10)

print("--- Moderate Favorite ---")
print_diff(0.50, 0.30, 0.20)

print("--- Slight Favorite ---")
print_diff(0.40, 0.30, 0.30)
