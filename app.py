import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array

# Load the Keras model
model = keras.models.load_model("weather_model.h5")

# UI
st.title("Image Classification for Weather")
st.write("Upload an image of a weather condition to classify it.")
st.write("The model can classify images into the following categories: cloudy, rain, shine, sunrise.")
st.warning("When uploading image, make sure to wait for at least 3 seconds before clicking 'Classify' to fully load the image and not result in error.")

# File upload
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if st.button("Classify"):
    if uploaded_file is not None:
        # Display image
        st.image(uploaded_file, caption="Uploaded Image")
        st.write("Classifying...")

        # Preprocess the image
        img = load_img(uploaded_file, target_size=(224, 224))  # fix target_size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # batch dimension

        # Predict
        predictions = model.predict(img_array)
        score = keras.activations.softmax(predictions[0]).numpy()

        # Class names (match your notebook!)
        classes = ['cloudy', 'rain', 'shine', 'sunrise']
        st.write(f"Prediction: **{classes[np.argmax(score)]}**")
    else:
        st.warning("Please upload an image.")
