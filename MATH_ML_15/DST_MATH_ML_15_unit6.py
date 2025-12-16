
import pandas as pd
import math
import numpy as np
from scipy.linalg import svd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

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
interactions_test_df = interactions_full_df.loc[interactions_full_df.last_timestamp >= split_ts].copy()

# --- Task 6.2 Logic ---
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)
ratings_matrix = users_items_pivot_matrix_df.values
global_mean = np.mean(ratings_matrix)
print(f"Global Mean (6.2): {global_mean:.3f}")

# --- Task 6.6 & 6.7 Logic ---
print("\n--- Task 6.6 & 6.7 ---")
# Decompose
U, sigma, Vt = svd(ratings_matrix) # full SVD
print(f"Max value in U: {np.max(U):.2f}")

k = 100
s = np.diag(sigma[:k])
U_k = U[:, 0:k]
Vt_k = Vt[0:k, :]

# Reconstruct the matrix
new_ratings_matrix = np.dot(np.dot(U_k, s), Vt_k)

# --- Task 6.8 Logic ---
print("\n--- Task 6.8 ---")

# Wrap reconstructed matrix in DataFrame for easier indexing
new_ratings_df = pd.DataFrame(new_ratings_matrix, 
                              index=users_items_pivot_matrix_df.index, 
                              columns=users_items_pivot_matrix_df.columns)

# Prepare test interactions
interactions = interactions_test_df.groupby('personId')['contentId'].agg(set).reset_index()
interactions.rename(columns={'contentId': 'true_test'}, inplace=True)

def make_prediction_svd(person_id):
    if person_id not in new_ratings_df.index:
        return []
    
    # Get user's predicted scores
    user_scores = new_ratings_df.loc[person_id]
    
    # Identify items user has already seen in training
    seen_mask = users_items_pivot_matrix_df.loc[person_id] > 0
    seen_articles = seen_mask[seen_mask].index
    
    # Exclude seen articles
    user_scores_excluded = user_scores.drop(index=seen_articles, errors='ignore')
    
    # Take top 10
    top_recommendations = user_scores_excluded.sort_values(ascending=False).head(10)
    
    return list(top_recommendations.index)

# Generate predictions
tqdm.pandas()
print("Generating SVD predictions...")
interactions['prediction_svd'] = interactions['personId'].progress_apply(make_prediction_svd)

# Calculate precision
def calc_precision(column, df):
    return (df.apply(lambda row: len(set(row['true_test']).intersection(
                set(row[column]))) / min(len(row['true_test']) + 0.001, 10.0), axis=1)).mean()

precision_svd = calc_precision('prediction_svd', interactions)
print(f"SVD Precision: {precision_svd}")
print(f"Rounded SVD Precision: {precision_svd:.3f}")

# --- Task 6.9 Logic ---
print("\n--- Task 6.9 ---")

# 1. Sparse Matrix
ratings_matrix_sparse = csr_matrix(users_items_pivot_matrix_df.values)

# 2. Split (30% validation, 70% train)
train_data, test_data = random_train_test_split(ratings_matrix_sparse, test_percentage=0.3, random_state=13)

# 3. Model
model = LightFM(learning_rate=0.05, loss='warp', no_components=100, random_state=13)

# 4. Train (epochs=10 is assumed standard if not specified, often sufficient for exercises)
model.fit(train_data, epochs=10)

# 5. Evaluate
precision_score = precision_at_k(model, test_data, k=10).mean()

print(f"LightFM Precision@10: {precision_score}")
print(f"Rounded LightFM Precision@10: {precision_score:.2f}")