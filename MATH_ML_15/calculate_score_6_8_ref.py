
import pandas as pd
import math
import numpy as np
from scipy.linalg import svd

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

interactions_df['eventStrength'] = interactions_df.eventType.apply(lambda x: event_type_strength[x])

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

def smooth_user_preference(x):
    return math.log(1+x, 2)

interactions_full_df = (
    interactions_from_selected_users_df
    .groupby(['personId', 'contentId']).eventStrength.sum()
    .apply(smooth_user_preference)
    .reset_index().set_index(['personId', 'contentId'])
)

interactions_full_df['last_timestamp'] = (
    interactions_from_selected_users_df
    .groupby(['personId', 'contentId'])['timestamp'].max()
)

interactions_full_df = interactions_full_df.reset_index()

split_ts = 1475519545
interactions_train_df = interactions_full_df.loc[interactions_full_df.last_timestamp < split_ts].copy()
interactions_test_df = interactions_full_df.loc[interactions_full_df.last_timestamp >= split_ts].copy()

# Pivot table (Train)
ratings = interactions_train_df.pivot(index='personId', 
                                      columns='contentId', 
                                      values='eventStrength').fillna(0)
ratings_matrix = ratings.values

# Decompose
U, sigma, V = svd(ratings_matrix)

# Reference Logic Task 6.7/6.8
k = 100
s = np.diag(sigma[:k])
U = U[:, 0:k]
V = V[0:k, :]

new_ratings = pd.DataFrame(
    U.dot(s).dot(V), index=ratings.index, columns=ratings.columns
)

# Prepare interactions DataFrame for Evaluation (matching reference structure)
# We need 'true_train' and 'true_test' columns in a DF indexed by personId
train_interactions = interactions_train_df.groupby('personId')['contentId'].agg(list) # Reference uses np.in1d which works with lists/arrays
test_interactions = interactions_test_df.groupby('personId')['contentId'].agg(list)

# We use users from TEST for evaluation
interactions = pd.DataFrame(index=test_interactions.index)
interactions['true_test'] = test_interactions
interactions['true_train'] = train_interactions

# CAUTION: Some users in Test might not be in Train.
# The reference code 'interactions.loc[personId, "true_train"]' would fail if NaN/key missing if not handled.
# Assuming standard pandas behavior, if we join, users not in Train get NaN for true_train.
# Let's fill NaNs with empty lists for robust logic.
interactions['true_train'] = interactions['true_train'].apply(lambda x: x if isinstance(x, list) else [])

# Define calc_precision EXACTLY as requested (global function tailored for the prediction_svd column logic)
def calc_precision(column):
    return ( interactions.apply(  lambda row:len(set(row['true_test']).intersection(
                set(row[column]))) /min(len(row['true_test']) + 0.001, 10.0), axis=1)).mean()

# Reference Loop
top_k = 10
predictions = []

print("Starting Reference Loop...")
for personId in interactions.index:
    # Handle Cold Start: User in Test but not in Train (new_ratings)
    if personId in new_ratings.index:
        prediction = (
            new_ratings.loc[personId].sort_values(ascending=False).index.values
        )
        
        # Filter seen
        # interactions.loc[personId, "true_train"] access
        # Since we are iterating personId from interactions.index, we can access row directly or via loc.
        # But 'interactions' variable inside loop is the global DF.
        train_items = interactions.loc[personId, "true_train"]
        
        # prediction is numpy array.
        # ~np.in1d(prediction, train_items) gives boolean mask
        
        predicted_items = list(
            prediction[
                ~np.in1d(prediction, train_items)
            ]
        )[:top_k]
        
        predictions.append(predicted_items)
    else:
        # User not in train set -> No predictions
        predictions.append([])

interactions["prediction_svd"] = predictions

precision = calc_precision("prediction_svd")
print(f"Reference Logic Precision: {precision}")
print(f"Rounded Reference Precision: {precision:.3f}")
