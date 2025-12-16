
import pandas as pd
import math
import numpy as np

# Load data
interactions_df = pd.read_csv('/home/pavel/IDE/skillfactory/MATH_ML_15/data/users_interactions.csv')

# Convert IDs to strings
interactions_df.personId = interactions_df.personId.astype(str)
interactions_df.contentId = interactions_df.contentId.astype(str)

# Define weights
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

# Apply weights
interactions_df['eventStrength'] = interactions_df.eventType.apply(lambda x: event_type_strength[x])

# Filter users with >= 5 interactions
users_interactions_count_df = (
    interactions_df
    .groupby(['personId', 'contentId'])
    .first()
    .reset_index()
    .groupby('personId').size())

users_with_enough_interactions_df = \
    users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]

interactions_from_selected_users_df = interactions_df.loc[np.in1d(interactions_df.personId,
            users_with_enough_interactions_df)]

# Smooth user preference function
def smooth_user_preference(x):
    return math.log(1+x, 2)

# Create full interaction dataframe
interactions_full_df = (
    interactions_from_selected_users_df
    .groupby(['personId', 'contentId']).eventStrength.sum()
    .apply(smooth_user_preference)
    .reset_index().set_index(['personId', 'contentId'])
)

# Add last_timestamp
interactions_full_df['last_timestamp'] = (
    interactions_from_selected_users_df
    .groupby(['personId', 'contentId'])['timestamp'].max()
)

interactions_full_df = interactions_full_df.reset_index()

# Train/Test logic
split_ts = 1475519545
interactions_train_df = interactions_full_df.loc[interactions_full_df.last_timestamp < split_ts].copy()

# Target User and Item
target_user = '-1032019229384696495'
target_item = '943818026930898372'

# Find the value
result_row = interactions_train_df[
    (interactions_train_df['personId'] == target_user) & 
    (interactions_train_df['contentId'] == target_item)
]

if not result_row.empty:
    score = result_row['eventStrength'].values[0]
    print(f"Score: {score}")
    print(f"Rounded Score: {score:.2f}")
else:
    print("Interaction not found in training set.")
