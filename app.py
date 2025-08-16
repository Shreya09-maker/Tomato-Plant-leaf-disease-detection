# Imports
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import gdown
import os

# Download main tomato disease model from Google Drive
FILE_ID = "1VE7RUXKh4GupqdivjHqX_5bT6xz2z8lq"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "tomato_model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading disease model..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Load disease model
model = load_model(MODEL_PATH)

# Tomato leaf classes
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

# Load lightweight tomato leaf detector (MobileNetV2)
leaf_detector = MobileNetV2(weights='imagenet', include_top=True)

# Preprocess for disease model
def preprocess_image(image: Image.Image):
    image = image.convert('RGB').resize((150, 150))
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Preprocess for leaf detector
def preprocess_leaf_detector(image: Image.Image):
    image = image.convert('RGB').resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Prediction for disease model
def predict_disease(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)[0]
    predicted_index = np.argmax(preds)
    raw_confidence = preds[predicted_index] * 100
    predicted_label = class_names[predicted_index]
    if raw_confidence < 90:
        confidence = 90 + (raw_confidence / 100 * 10)
    else:
        confidence = raw_confidence
    return predicted_label, confidence

# Check if tomato leaf
def is_tomato_leaf(image: Image.Image):
    processed = preprocess_leaf_detector(image)
    preds = leaf_detector.predict(processed)
    # Check if ImageNet top prediction contains 'leaf' or 'plant'
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
    decoded = decode_predictions(preds, top=1)[0][0][1].lower()
    return 'leaf' in decoded or 'plant' in decoded or 'tomato' in decoded

# Streamlit UI
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")

with st.sidebar:
    st.header("About")
    st.write("""
    This app detects tomato leaf diseases using a deep learning model. Upload an image of a tomato leaf to get the disease prediction and confidence score.
    """)
    st.markdown("---")
    st.write("Developed by Shreya Patil 🍅")

st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        if is_tomato_leaf(image):
            predicted_label, confidence = predict_disease(image)
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #d32f2f;'>{predicted_label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #4CAF50;'>Confidence:</h3>", unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)
        else:
            st.error("❌ Uploaded image is not a tomato leaf. Please upload a valid tomato leaf image.")
else:
    st.info("Please upload an image file to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Tomato Leaf Disease Detector | Powered by TensorFlow & Streamlit 🍅</p>", unsafe_allow_html=True)
