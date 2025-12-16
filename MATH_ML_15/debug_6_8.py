
import pandas as pd
import math
import numpy as np
from scipy.linalg import svd

# Load data
interactions_df = pd.read_csv('/home/pavel/IDE/skillfactory/MATH_ML_15/data/users_interactions.csv')

# Convert IDs to strings
interactions_df.personId = interactions_df.personId.astype(str)
interactions_df.contentId = interactions_df.contentId.astype(str)

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

# Pivot
ratings = interactions_train_df.pivot(index='personId', 
                                      columns='contentId', 
                                      values='eventStrength').fillna(0)
ratings_matrix = ratings.values

# Decompose
U, sigma, V = svd(ratings_matrix) # Dense SVD

# k=100
k = 100
s = np.diag(sigma[:k])
U_k = U[:, 0:k]
V_k = V[0:k, :]

new_ratings = pd.DataFrame(
    U_k.dot(s).dot(V_k), index=ratings.index, columns=ratings.columns
)

# Eval
train_interactions = interactions_train_df.groupby('personId')['contentId'].agg(list)
test_interactions = interactions_test_df.groupby('personId')['contentId'].agg(list)

interactions = pd.DataFrame(index=test_interactions.index)
interactions['true_test'] = test_interactions
# Join train interactions, filling missing with empty lists
interactions['true_train'] = train_interactions
interactions['true_train'] = interactions['true_train'].fillna("").apply(list) # fillna("") makes it iterable string if used? No, safe to use apply.
# Correct way to fill NaN with empty list:
for idx in interactions.index:
    if not isinstance(interactions.at[idx, 'true_train'], list):
         interactions.at[idx, 'true_train'] = []

top_k = 8
predictions = []

hits = 0
total_users = 0

for personId in interactions.index:
    total_users += 1
    if personId in new_ratings.index:
        # User in Train
        user_row = new_ratings.loc[personId]
        # Sort desc
        sorted_items = user_row.sort_values(ascending=False).index.values
        
        # Filter seen
        seen_items = interactions.at[personId, "true_train"]
        
        # optimized exclusion
        # mask = ~np.in1d(sorted_items, seen_items)
        # candidates = sorted_items[mask][:top_k]
        
        candidates = []
        for item in sorted_items:
            if item not in seen_items:
                candidates.append(item)
            if len(candidates) == top_k:
                break
        
        predictions.append(candidates)
    else:
        predictions.append([])

interactions["prediction_svd"] = predictions

def calc_precision(column):
    return ( interactions.apply(  lambda row:len(set(row['true_test']).intersection(
                set(row[column]))) /min(len(row['true_test']) + 0.001, 10.0), axis=1)).mean()

# Debug stats
users_in_train = interactions.index.isin(new_ratings.index).sum()
print(f"Total Test Users: {len(interactions)}")
print(f"Test Users in Train: {users_in_train}")

precision = calc_precision("prediction_svd")
print(f"Precision: {precision}")
