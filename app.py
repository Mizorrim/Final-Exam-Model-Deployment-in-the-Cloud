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
st.warning("When uploading image, make sure to wait for at least 3 seconds before clicking 'Classify'".)

# File upload
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        st.write("Classifying...")

        try:
            # Preprocess the image
            img = load_img(uploaded_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_array)
            score = keras.activations.softmax(predictions[0]).numpy()

            classes = ['cloudy', 'rain', 'shine', 'sunrise']
            st.write(f"Prediction: **{classes[np.argmax(score)]}**")
        except Exception as e:
            st.error(f"Something went wrong during classification: {e}")
else:
    st.warning("Please upload an image.")

