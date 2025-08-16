# generate_feature_from_dataset.py

from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# ✅ STEP 1: SET CORRECT PATH (edit this if needed)
DATASET_FOLDER = r"C:\PlantVillage"  # Move your folder here for simplicity

# ✅ STEP 2: CHECK IF FOLDER EXISTS
print(f"Checking dataset path: {DATASET_FOLDER}")
if not os.path.exists(DATASET_FOLDER):
    raise FileNotFoundError(f"❌ Dataset folder not found: {DATASET_FOLDER}")
else:
    print("✅ Folder found. Extracting images...")

# ✅ STEP 3: COLLECT IMAGES
list_of_images = []
for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            list_of_images.append(os.path.join(root, file))

if len(list_of_images) == 0:
    raise ValueError(f"❌ No images found in {DATASET_FOLDER}")

# ✅ STEP 4: LOAD MODEL
leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# ✅ STEP 5: EXTRACT FEATURES
features_list = []
for path in list_of_images:
    img = Image.open(path).convert('RGB').resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = leaf_detector.predict(x)[0]
    features_list.append(features)

# ✅ STEP 6: AVERAGE AND SAVE FEATURE VECTOR
tomato_leaf_feature = np.mean(features_list, axis=0)
np.save("tomato_leaf_feature.npy", tomato_leaf_feature)
print("✅ tomato_leaf_feature.npy saved successfully.")
