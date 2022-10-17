import streamlit as st
import cv2
import time
import os
import tempfile
import matplotlib.pyplot as plt
from src.utils.onnx_process import load_model, video_predict
from src.utils.video_process import video_stitch
from src.utils.streamlit import save_uploaded_file

MODEL_PATH = "./results/models/onnx_dive/model.onnx"
LABEL_PATH = "./results/models/onnx_dive/label_map.pbtxt"
MODEL_INPUT_SIZE = (640, 640)  # width, height
NUM_CLASSES = 3
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.1
# BBOX_VIDEO_NAME = "bbox_video"

##STEP 1 Load Model
session, input_name, output_name = load_model(MODEL_PATH)

##STEP 2 Upload Video
st.write("# 2. Upload raw diving video:\n")

# create temp dir for storing video and outputs
temp_dir = tempfile.TemporaryDirectory()
temp_path = temp_dir.name

video_file = st.file_uploader(
    "Choose a File", accept_multiple_files=False, type=["mp4", "mov"]
)

# st.video(vid_file)

if video_file is not None:
    file_details = {"FileName": video_file.name, "FileType": video_file.type}
    st.write(file_details)
    video_path = save_uploaded_file(video_file, temp_path)
    st.write(video_path)

    trim_bt = st.button("Start Auto-Trimming!")
    st.write(trim_bt)
    if trim_bt:
        with st.spinner(text="YOEO working its magic: IN PROGRESS ..."):
            (
                frame_predictions,
                bbox_class_score,
                orig_frames,
                origi_shape,
                fps,
            ) = video_predict(
                video_path,
                "frames",
                session,
                input_name,
                output_name,
                LABEL_PATH,
                MODEL_INPUT_SIZE,
                NUM_CLASSES,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
            )

        bbox_video_path = os.path.join(temp_path, "orig_video")

        video_stitch(
            frame_predictions,
            bbox_video_path,
            video_file.name.replace(".mp4", ""),
            origi_shape,
            fps,
        )

        # recode video using ffmpeg
        video_bbox_filename = os.path.join(bbox_video_path, video_file.name)
        video_bbox_recode_filename = video_bbox_filename.replace(".mp4", "_recoded.mp4")
        os.system(
            "ffmpeg -i {} -vcodec libx264 {}".format(
                os.path.join(bbox_video_path, video_file.name),
                video_bbox_recode_filename,
            )
        )

        st.write(video_bbox_filename)
        # st.write(os.listdir(os.path.join(RESULTS_PATH, latest_folder)))
        st.write(video_bbox_recode_filename)
        st.write("YOEO's Object Detection Results:")
        st.video(video_bbox_recode_filename)

##STEP 3
st.write("# 3. YOEO working its magic: ")
st.write("-> to insert model inference and stich algo in progress bar")
my_bar = st.progress(0)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)


##STEP 4
st.write("# 4. Objects of interest detected and trimmed video output: ")

col1, col2, col3 = st.columns(3)
col1.metric("# Species Detected", "2")
col2.metric("Turtle", "1")
col3.metric("Fish", "23")

# st.video(vid_file)
