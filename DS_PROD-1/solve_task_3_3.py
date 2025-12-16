
import pickle
import sys

# 1. Load the model from 'model.pkl'
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Extract attributes 'a' and 'b' and create the dictionary
# Note: The model object might be of a custom class.
# We access a and b directly as attributes.
try:
    data_dict = {'a': model.a, 'b': model.b}
    print(f"Extracted dictionary: {data_dict}")
except AttributeError:
    print("Error: The model does not have attributes 'a' and 'b'.")
    # Let's inspect the model dir just in case
    print(f"Model attributes: {dir(model)}")
    sys.exit(1)

# 3. Save the dictionary to a new pickle file
output_filename = 'solution_3_3.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Dictionary saved to {output_filename}")
