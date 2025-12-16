import scipy.stats as st
import math

n = 189
positive = 132
negative = n - positive

# 1. Sample proportion
p_hat = positive / n
print(f"Sample proportion (p_hat): {p_hat:.3f}")

# 2. 90% Confidence Interval
gamma = 0.90
alpha = 1 - gamma
z_crit = -st.norm.ppf(alpha/2) # or st.norm.ppf(1 - alpha/2)

print(f"Z-critical: {z_crit:.4f}")

margin_of_error = z_crit * math.sqrt((p_hat * (1 - p_hat)) / n)

lower_bound = (p_hat - margin_of_error) * 100
upper_bound = (p_hat + margin_of_error) * 100

print(f"90% Confidence Interval: [{lower_bound:.2f}%, {upper_bound:.2f}%]")
