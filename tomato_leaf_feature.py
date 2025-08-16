import numpy as np

# Load the .npy file
features = np.load('tomato_leaf_feature.npy')

# Print the type and shape of the data
print(f"Type of data: {type(features)}")
print(f"Shape of data: {features.shape}")

# Optionally, print the first few elements to get a sense of the data
print("First few entries:")
print(features[:5])
