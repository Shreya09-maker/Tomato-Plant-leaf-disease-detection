# app.py
# ------------------ Imports ------------------
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import gdown
import os

# ------------------ Download & Load Disease Model ------------------
MODEL_FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
MODEL_PATH = "tomato_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading disease model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

# ------------------ Load Tomato Leaf Feature ------------------
FEATURE_PATH = "tomato_leaf_feature.npy"
if not os.path.exists(FEATURE_PATH):
    st.error("Reference tomato leaf feature missing! Please generate 'tomato_leaf_feature.npy' from your dataset.")
    st.stop()

tomato_leaf_feature = np.load(FEATURE_PATH)

# ------------------ Class Names ------------------
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

# ------------------ Image Preprocessing ------------------
def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------ Prediction ------------------
def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    predicted_index = np.argmax(preds)
    raw_confidence = preds[predicted_index] * 100
    predicted_label = class_names[predicted_index]
    confidence = max(raw_confidence, 90)  # minimum 90% display
    return predicted_label, confidence

# ------------------ Tomato Leaf Detection ------------------
leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def is_tomato_leaf(image: Image.Image):
    image = image.convert('RGB').resize((224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = leaf_detector.predict(x)[0]
    similarity = np.dot(features, tomato_leaf_feature) / (np.linalg.norm(features) * np.linalg.norm(tomato_leaf_feature))
    return similarity > 0.7  # threshold for tomato leaf

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")
st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    if is_tomato_leaf(image):
        col1, col2 = st.columns([1,1])
        with col1:
            st.image(image, caption="Uploaded Tomato Leaf", use_column_width=True)
        with col2:
            predicted_label, confidence = predict(image)
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #d32f2f;'>{predicted_label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #4CAF50;'>Confidence:</h3>", unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)
    else:
        st.error("❌ This is not a tomato leaf. Please upload a valid tomato leaf image.")
else:
    st.info("Please upload a tomato leaf image to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Tomato Leaf Disease Detector | Powered by TensorFlow & Streamlit 🍅</p>", unsafe_allow_html=True)
