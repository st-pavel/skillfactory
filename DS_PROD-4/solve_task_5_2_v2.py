import pandas as pd
import os
from statsmodels.stats.proportion import proportions_ztest

# Define paths
data_dir = '/home/pavel/IDE/skillfactory/DS_PROD-4/data'
file_a = os.path.join(data_dir, 'ab_test-redesign_sample_a.zip')
file_b = os.path.join(data_dir, 'ab_test-redesign_sample_b.zip')

# Read data
sample_a = pd.read_csv(file_a)
sample_b = pd.read_csv(file_b)

# successes
successes_a = sample_a['transactions'].sum()
nobs_a = sample_a['cid'].count()

successes_b = sample_b['transactions'].sum()
nobs_b = sample_b['cid'].count()

print(f"Group A: Successes={successes_a}, Nobs={nobs_a}, Prop={successes_a/nobs_a:.4f}")
print(f"Group B: Successes={successes_b}, Nobs={nobs_b}, Prop={successes_b/nobs_b:.4f}")

# Perform Z-test with alternative='smaller'
# H0: p_a >= p_b
# H1: p_a < p_b
alpha = 0.05
stat, p_value = proportions_ztest(count=[successes_a, successes_b], nobs=[nobs_a, nobs_b], alternative='smaller')

print(f"Z-statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < alpha:
    print("Reject Null Hypothesis (p_a >= p_b).")
    print("Conclusion: Proportion in Group A is significantly smaller than in Group B.")
    print("Variant B is better.")
else:
    print("Fail to reject Null Hypothesis.")
    print("Conclusion: There is no statistical evidence that Group A is worse than Group B (or B is better than A).")
    print("Variants are equivalent (or A is better, but we know A is numerically smaller).")
