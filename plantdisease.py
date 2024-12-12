import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
try:
    model = tf.keras.models.load_model("model.h5")
except Exception as e:
    st.error(f"Error loading the model: {e}")
print(model.input_shape)

# Class labels
class_names = [
    "APPLE SCAB",
    "GRAPE ESCA BLACK MEASLES",
    "MAIZE RUST",
    "ORANGE CITRUS GREENING",
    "PEPPER BALL BACTERIAL SPOT",
    "SOYBEAN HEALTHY",
    "TOMATO YELLOW LEAF CURL VIRUS",
]

# Streamlit app layout
st.title("Plant Disease Classification")
st.write("Upload an image of a plant leaf to predict the disease.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image for the model
        image = image.resize((150, 150))  # Resize to 150x150
        image_array = np.array(image) / 255.0  # Normalize to range [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions, axis=1)[
            0
        ]  # Get highest prediction index
        predicted_label = class_names[predicted_class]

        # Display the prediction
        st.success(f"Predicted Disease: {predicted_label}")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.info("Please upload an image to get started.")
