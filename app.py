import streamlit as st
from PIL import Image
import tempfile
import numpy as np
from io import BytesIO

st.title(" Drone vs  Bird Detection")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)


model = YOLO(best_model)

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        results = model.predict(source=temp_path, conf=conf_threshold)

        result_img = results[0].plot()
        st.image(result_img, caption="Detection Result", use_column_width=True)

        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                st.write(f"{label} - Confidence: {conf:.2f}")

        result_array = np.array(result_img)
        result_pil = Image.fromarray(result_array)

        buf = BytesIO()
        result_pil.save(buf, format="JPEG")

        st.download_button(
            label="Download Result",
            data=buf.getvalue(),
            file_name=f"detection_{i}.jpg",
            mime="image/jpeg",
            key=f"download_{i}"
        )
