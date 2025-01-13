import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the trained model
model = load_model("material_classifier.h5")

# Class indices mapping (update with your actual class names)
class_names = {0: "Cardboard", 1: "Plastic", 2: "Metal", 3: "Glass"}

# Function to preprocess image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app
st.title("Material Classifier")

st.write("Upload an image to classify it as Cardboard, Plastic, Metal, or Glass.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Process the uploaded image
    from PIL import Image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image, target_size=(224, 224))

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index]

    # Display prediction
    st.subheader("Prediction Results")
    st.write(f"Class: **{predicted_class_name}**")
    st.write(f"Confidence Score: **{confidence_score:.2f}**")

    # Provide additional information based on the predicted class
    if predicted_class_name == "Cardboard":
        st.info("Cardboard can be recycled to make new boxes and paper products.")
    elif predicted_class_name == "Plastic":
        st.warning("Plastic recycling is essential to reduce ocean pollution and landfill waste.")
    elif predicted_class_name == "Metal":
        st.info("Metal recycling saves energy and resources compared to extracting new metals.")
    elif predicted_class_name == "Glass":
        st.info("Glass can be melted down and reused indefinitely without quality loss.")
