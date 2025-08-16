import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load your trained model
model = load_model("tomato_model.h5")

# Class names (change according to your model)
class_names = [
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "Tomato healthy"
]

# Function to preprocess image for model
def preprocess(image):
    img = image.resize((224, 224))  # resize to model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ Function to check if image is a leaf
def is_leaf(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Green color range
    lower = np.array([25, 40, 40])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(img, lower, upper)
    green_ratio = np.sum(mask > 0) / mask.size

    return green_ratio > 0.05  # at least 5% green pixels required

# Streamlit UI
st.title("🌱 Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ✅ Step 1: Check if it is a leaf
    if not is_leaf(image):
        st.error("⚠️ Please upload a valid tomato leaf image!")
    else:
        # ✅ Step 2: Predict with model
        img_array = preprocess(image)
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.markdown(f"**Prediction:** <span style='color:red;'>{predicted_class}</span>", unsafe_allow_html=True)
        st.markdown("**Confidence:**")
        st.progress(float(confidence))
        st.write(f"{confidence*100:.2f}% confident")
