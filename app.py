import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.title("YOLO Object Detection")

# Load model safely
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Make sure best.pt is uploaded.")
    st.stop()

model = YOLO(MODEL_PATH)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # Run YOLO
    results = model(img_array)

    # Plot results
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Image", use_column_width=True)
