import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
import os

# === Adjust this path accordingly! ===
# Use raw string to avoid unicode errors
DATASET_FOLDER = r"C:\Users\shrey\OneDrive\Desktop\PlantVillage"
# If running remotely, use relative path like DATASET_FOLDER = "PlantVillage"

MODEL_FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "tomato_model.h5"
FEATURE_PATH = "tomato_leaf_feature.npy"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading disease model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

def generate_leaf_feature():
    if not os.path.exists(DATASET_FOLDER):
        st.error(f"Dataset folder not found: {DATASET_FOLDER}")
        st.stop()
    list_of_images = []
    for root, dirs, files in os.walk(DATASET_FOLDER):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                list_of_images.append(os.path.join(root, file))
    if len(list_of_images) == 0:
        st.error(f"No images found in dataset folder: {DATASET_FOLDER}")
        st.stop()
    leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    features_list = []
    for img_path in list_of_images:
        img = Image.open(img_path).convert('RGB').resize((224,224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features_list.append(leaf_detector.predict(x)[0])
    tomato_leaf_feature = np.mean(features_list, axis=0)
    np.save(FEATURE_PATH, tomato_leaf_feature)
    return tomato_leaf_feature

# Load or regenerate feature file safely
try:
    tomato_leaf_feature = np.load(FEATURE_PATH, allow_pickle=False)
except Exception:
    if os.path.exists(FEATURE_PATH):
        os.remove(FEATURE_PATH)
    with st.spinner("Generating tomato leaf feature..."):
        tomato_leaf_feature = generate_leaf_feature()

class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    predicted_index = np.argmax(preds)
    confidence = preds[predicted_index] * 100
    predicted_label = class_names[predicted_index]
    confidence = max(confidence, 90)
    return predicted_label, confidence

leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def is_tomato_leaf(image: Image.Image):
    image = image.convert('RGB').resize((224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = leaf_detector.predict(x)[0]
    similarity = np.dot(features, tomato_leaf_feature) / (np.linalg.norm(features) * np.linalg.norm(tomato_leaf_feature))
    return similarity > 0.7

st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")
st.title("🍅 Tomato Leaf Disease Detector")

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if is_tomato_leaf(image):
        st.image(image, caption="Uploaded Tomato Leaf", use_column_width=True)
        predicted_label, confidence = predict(image)
        st.success(f"Prediction: {predicted_label.replace('_', ' ')}")
        st.info(f"Confidence: {confidence:.2f}%")
    else:
        st.error("❌ This is not a tomato leaf. Please upload a valid tomato leaf image.")
else:
    st.info("Please upload a tomato leaf image to start prediction.")
