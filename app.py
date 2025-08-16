import uuid
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load MobileNetV2 feature extractor (no top) for tomato leaf detection
leaf_detector = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Example: precomputed tomato leaf feature (replace with real features for accuracy)
tomato_leaf_feature = np.random.rand(1280)  # MobileNetV2 avg pooling output size

# Fixed tomato leaf check
def is_tomato_leaf(image):
    image = image.convert('RGB').resize((224, 224))
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = leaf_detector.predict(x)[0]
    
    # Cosine similarity
    similarity = np.dot(features, tomato_leaf_feature) / (np.linalg.norm(features) * np.linalg.norm(tomato_leaf_feature))
    return similarity > 0.7  # threshold for tomato leaf

# Generate a unique ID for each uploaded image
def generate_upload_id():
    return str(uuid.uuid4())

# Example usage in your existing app:
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    upload_id = generate_upload_id()
    st.text(f"Upload ID: {upload_id}")  # Display unique ID

    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        if is_tomato_leaf(image):
            predicted_label, confidence = predict(image)
            st.markdown(f"<h3 style='color: #4CAF50;'>Prediction:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color: #d32f2f;'>{predicted_label.replace('_', ' ')}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #4CAF50;'>Confidence:</h3>", unsafe_allow_html=True)
            st.progress(min(int(confidence), 100))
            st.markdown(f"<h4 style='color: #555;'>{confidence:.2f}% confident</h4>", unsafe_allow_html=True)
        else:
            st.error("❌ Uploaded image is not a tomato leaf. Please upload a valid tomato leaf image.")
