import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained models (make sure both files are in the same folder as app.py)

DISEASE_MODEL_PATH = "tomato_model.h5"

leaf_detector = load_model(LEAF_MODEL_PATH)
disease_model = load_model(DISEASE_MODEL_PATH)

# Class names for disease prediction
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

def preprocess_image(image: Image.Image, size=(150, 150)):
    """Resize and normalize image for model input."""
    image = image.convert('RGB').resize(size)
    img_array = img_to_array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_leaf(image: Image.Image):
    """Predict if the image is a tomato leaf (binary classification)."""
    processed = preprocess_image(image)
    pred = leaf_detector.predict(processed)[0][0]
    return pred > 0.5  # True if tomato leaf, False if not

def predict_disease(image: Image.Image):
    """Predict tomato leaf disease class and confidence."""
    processed = preprocess_image(image)
    preds = disease_model.predict(processed)[0]
    idx = np.argmax(preds)
    confidence = preds[idx] * 100
    label = class_names[idx]
    return label, confidence

# Streamlit app setup
st.set_page_config(page_title="🍅 Tomato Leaf Disease Detector", layout="wide", page_icon="🍅")

with st.sidebar:
    st.header("About")
    st.write("""
        This app first checks if the uploaded image is a tomato leaf, 
        then predicts the disease using a deep learning model.
        
        **Class Labels:**
        - Bacterial Spot
        - Early Blight
        - Late Blight
        - Leaf Mold
        - Septoria Leaf Spot
        - Spider Mites (Two-spotted)
        - Target Spot
        - Tomato Yellow Leaf Curl Virus
        - Tomato Mosaic Virus
        - Healthy
    """)
    st.markdown("---")
    st.write("Developed by Shreya Patil 🍅")

st.markdown("<h1 style='text-align: center; color: green;'>🍅 Tomato Leaf Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image of a tomato leaf below to get started.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Step 1: Check if image is tomato leaf
    if not predict_leaf(image):
        st.error("❌ This image does not look like a tomato leaf. Please upload a valid tomato leaf image.")
    else:
        # Step 2: Predict disease
        label, confidence = predict_disease(image)
        confidence_threshold = 60  # Set your threshold

        if confidence < confidence_threshold:
            st.error("❌ Low confidence in prediction. Please try another image.")
        else:
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(image, caption="Uploaded Tomato Leaf Image", use_container_width=True)
            with col2:
                st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color: #d32f2f;'>{label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: #4CAF50;'>Confidence:</h3>", unsafe_allow_html=True)
                st.progress(min(int(confidence), 100))
                st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)

else:
    st.info("Please upload an image file to start prediction.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2025 Tomato Leaf Disease Detector | Powered by TensorFlow & Streamlit 🍅</p>", unsafe_allow_html=True)


