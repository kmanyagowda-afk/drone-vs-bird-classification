from io import BytesIO
import streamlit as st
from PIL import Image
import tempfile
import numpy as np

st.title(" Drone vs  Bird Detection")

conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name


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
