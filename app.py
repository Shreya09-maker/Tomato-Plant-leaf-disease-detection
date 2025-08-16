# Imports
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import uuid
import gdown

# ------------------ Download and Load Disease Model ------------------
FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"  # Google Drive file ID
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "tomato_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading disease model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

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

# ------------------ Preprocessing & Prediction ------------------
def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    predicted_index = np.argmax(preds)
    predicted_label = class_names[predicted_index]
    confidence = preds[predicted_index] * 100
    return predicted_label, confidence

# ------------------ Tomato Leaf Detection ------------------
leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

FEATURE_FILE = "tomato_leaf_feature.npy"

if not os.path.exists(FEATURE_FILE):
    st.error("Reference tomato leaf feature missing. Add 'tomato_leaf_feature.npy' in the app folder.")
    st.stop()

tomato_leaf_feature = np.load(FEATURE_FILE)

def is_tomato_leaf(image: Image.Image):
    image = image.convert('RGB').resize((224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = leaf_detector.predict(x)[0]
    similarity = np.dot(features, tomato_leaf_feature) / (np.linalg.norm(features) * np.linalg.norm(tomato_leaf_feature))
    return similarity > 0.7  # Only accept tomato leaf

# ------------------ Dataset vs Live Detection ------------------
DATASET_FOLDER = "PlantVillage/Tomato"  # Adjust your dataset path

def detect_image_source(image: Image.Image):
    uploaded_array = np.array(image.convert('RGB').resize((150, 150)))
    for root, dirs, files in os.walk(DATASET_FOLDER):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                try:
                    dataset_image = Image.open(os.path.join(root, file)).convert('RGB').resize((150, 150))
                    dataset_array = np.array(dataset_image)
                    if np.array_equal(uploaded_array, dataset_array):
                        return "Dataset"
                except:
                    continue
    return "Live"

# ------------------ Generate Unique Upload ID ------------------
def generate_upload_id():
    return str(uuid.uuid4())

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")
st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    upload_id = generate_upload_id()
    st.text(f"Upload ID: {upload_id}")

    if is_tomato_leaf(image):
        source = detect_image_source(image)
        st.image(image, caption=f"Uploaded Tomato Leaf ({source})", use_container_width=True)
        predicted_label, confidence = predict(image)
        st.markdown(f"### Prediction: **{predicted_label.replace('_', ' ')}**")
        st.progress(min(int(confidence), 100))
        s
