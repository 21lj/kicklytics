import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import random

# Load model once
@st.cache_resource
def load_model():
    return YOLO("../model/best.pt")

model = load_model()

st.title("YOLO Object Detection")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:

    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(uploaded_file.read())
        input_path = f.name

    st.video(input_path)

    if st.button("Run Detection"):

        with st.spinner("Running detection..."):
            nos = [random.randint(10000, 99999) for _ in range(10)]
            nos= ''.join(map(str, nos))
            results = model.predict(
                input_path,
                save=True,
                project=r"D:\RIT\projects\mini_project\backend\output_videos\objects_detect",
                name=f"objects_detect_result_{nos}",
                exist_ok=True
            )

            output_dir = results[0].save_dir

            st.success("Detection completed!")

            # Find generated video
            video_files = [
                file for file in os.listdir(output_dir)
                if file.endswith((".mp4", ".avi", ".mov"))
            ]

            if video_files:

                output_path = os.path.join(
                    output_dir,
                    video_files[0]
                )

                st.write("Output video:", output_path)

                if os.path.exists(output_path):
                    st.video(output_path)
                else:
                    st.error("Output file does not exist")

            else:
                st.error("No output video found")