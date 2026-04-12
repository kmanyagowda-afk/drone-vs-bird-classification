from io import BytesIO
import streamlit as st
from PIL import Image
import tempfile
import numpy as np

st.title(" Drone vs  Bird Detection")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):

        image = Image.open(uploaded_file).convert("RGB")
        

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

confidence_threshold = st.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.25
)

results = model(temp_path, conf=confidence_threshold)
result = results[0]
boxes = result.boxes

for box in boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    label = model.names[cls]
    st.write(f"{label} - Confidence: {conf:.2f}")

result_img = result.plot()   # <-- ADD THIS LINE
result_array = np.array(result_img)
result_pil = Image.fromarray(result_array)
st.image(result_pil)

     # create buffer properly
buf = BytesIO()
result_pil.save(buf, format="JPEG")
buf.seek(0)

st.download_button(
    label="Download Result",
    data=buf,
    file_name="detection.jpg",
    mime="image/jpeg"
)
