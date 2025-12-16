import pandas as pd
import os

# Set path to data
data_path = './data/ab_data.zip'

# Load data
ab_data = pd.read_csv(data_path)

# Convert timestamp to datetime
ab_data['timestamp'] = pd.to_datetime(ab_data['timestamp'], format='%Y-%m-%d')

# Group data
daily_data = ab_data.groupby(['timestamp','group']).agg({
    'user_id':'count',
    'converted':'sum'
}).reset_index().rename(columns={'user_id': 'users_count'})

# Calculate conversion
daily_data['conversion'] = (daily_data['converted'] / daily_data['users_count']) * 100

# Filter for Group A on 2017-01-05
target_date = pd.to_datetime('2017-01-05')
result = daily_data[(daily_data['timestamp'] == target_date) & (daily_data['group'] == 'A')]

print("Conversion for Group A on 2017-01-05:")
print(result[['timestamp', 'group', 'conversion']])
print(f"Rounded result: {result['conversion'].iloc[0]:.2f}")
