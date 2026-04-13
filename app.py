import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Aerial Object Detection")

# Load your .h5 model
model = tf.keras.models.load_model("best_aerial.h5")

uploaded_file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224,224))  # adjust if needed
    st.image(image, caption="Uploaded Image")

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    st.write("Prediction:", prediction)
