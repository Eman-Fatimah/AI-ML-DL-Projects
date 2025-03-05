import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Preprocess the image
def preprocess_image(image, target_size):
    # Resize the image to the target size
    image = image.resize(target_size)
    
    # Convert to RGB to ensure 3 channels
    image = image.convert("RGB")
    
    # Normalize pixel values to [0, 1]
    # image = tf.cast(np.array(image) / 255.0, tf.float32)

    # # Add batch dimension
    # image = tf.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image


# Load the trained model
model = tf.keras.models.load_model("model.keras")

# Fetch model's required input shape
input_shape = model.input_shape[1:3]  # (height, width)

# Streamlit app setup
st.title("Waste Segregation Model")

st.write("Upload an image of the waste, and the model will classify it as glass, plastic, trash, cupboard, paper, or metal..")

# File uploader for image input
uploaded_file = st.file_uploader(
    "Upload an image", 
    type=["jpeg", "png", "jpg"],
    accept_multiple_files=False
)

if st.button("Classify waste"):
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        st.write("Preprocessing the image...")
        processed_image = preprocess_image(image, input_shape)
        st.write(f"Image preprocessed. Shape: {processed_image.shape}")

        try:
            # Model prediction
            st.write("Model is processing the image...")
            prediction = model(processed_image)  # Get prediction probabilities
            predicted_label = tf.argmax(prediction, axis=-1).numpy()[0]  # Get the class with highest probability

            # # Define class labels
            class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

            # Display prediction result
            predicted_class = class_labels[predicted_label]
            st.success(f"The model predicts: {predicted_class}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please upload an image before clicking Classify Waste.")


