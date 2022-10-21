import streamlit as st
import cv2
import os
import sys
import tempfile
import shutil
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils.streamlit import save_uploaded_file, factors
from src.utils.yolox_process import video_predict
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


def load_model(ckpt_file, depth=0.33, width=0.25, num_classes=5):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    # NANO model use depthwise = True, which is main difference.
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, depthwise=True)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, depthwise=True)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    model.eval()
    # model

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    return model


MODEL_PATH = "./results/models/yolox_dive.pth"
MODEL_NAME = "yolox-nano"
EXP_PATH = "./exps/example/custom/nano.py"
MODEL_INPUT_SIZE = (640, 640)  # width, height
NUM_CLASSES = 5
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
YOEO_CLASSES = (
    "shark",
    "coral",
    "fish",
    "turtle",
    "manta ray",
)


##STEP 1 Load Model
with st.spinner(text="Loading Model ... Please be patient!"):
    model = load_model(MODEL_PATH)

##STEP 2 Upload Video
st.write("# Upload diving video:\n")

with st.expander("How to Use YOEO"):
    st.write("............")


# create temp dir for storing video and outputs
temp_dir = tempfile.TemporaryDirectory()
temp_path = temp_dir.name

video_file = st.file_uploader(
    "Choose a File", accept_multiple_files=False, type=["mp4", "mov"]
)

if video_file is not None:
    file_details = {"FileName": video_file.name, "FileType": video_file.type}
    st.write(file_details)
    video_path = save_uploaded_file(video_file, temp_path)
    st.write(video_path)
    video_bbox_filename = os.path.join(temp_path, video_file.name)
    # get fps for optimization slider max value
    fps = round(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    factors_fps = list(factors(fps))

    # user options
    marine_options = st.multiselect(
        "What flora & fauna do you prefer",
        ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
        ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
        help="Select the flora & fauna you want to be included in the final video",
    )
    marine_options_idx = [0, 1, 2, 3, 4]

    # user advanced options
    with st.expander("Advanced Options"):
        with st.form("my_form"):
            st.write("###### Leave as default if unsure!")
            ifps = st.select_slider(
                "Optimization", options=factors_fps, value=max(factors_fps)
            )  # num of frames per sec to do inferencing
            strict_val = st.slider(
                "Trimming Strictness", min_value=0, value=fps
            )  # number of frames prior to keep if current frame is to be kept
            sharpen = st.checkbox("Sharpen Video")
            color_grade = st.checkbox("Color Grade Video")
            yt_link = st.text_input("Enter a Youtube Audio Link")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit Advanced Options")
            if submitted:
                st.write("Optimization", ifps, "strict_val", strict_val)

    # start inferencing
    trim_bt = st.button("Start Auto-Trimming!")
    st.write(trim_bt)
    if trim_bt:
        with st.spinner(text="YOEO working its magic: IN PROGRESS ..."):
            bbox_class_score, orig_frames, origi_shape, fps = video_predict(
                video_path,
                video_bbox_filename,
                model,
                NUM_CLASSES,
                CONF_THRESHOLD,
                NMS_THRESHOLD,
                MODEL_INPUT_SIZE,
                YOEO_CLASSES,
                ifps,
                verbose=True,
            )

        # recode video using ffmpeg
        video_bbox_recode_filename = video_bbox_filename.replace(".mp4", "_recoded.mp4")
        os.system(
            "ffmpeg -i {} -vcodec libx264 {}".format(
                os.path.join(video_bbox_filename),
                video_bbox_recode_filename,
            )
        )

        # remove original video and obj detect video to save space
        os.remove(video_bbox_filename)
        shutil.rmtree(os.path.join(temp_path, "orig_video"))
        st.write(os.listdir(temp_path))

        tab_od, tab_trim, tab_beauty = st.tabs(
            [
                "YOEO's Object Detection Results",
                "Your Trimmed Video",
                "Beautiful Photos Captured By You",
            ]
        )
        with tab_od:
            st.write(video_bbox_filename)
            # st.write(os.listdir(os.path.join(RESULTS_PATH, latest_folder)))
            st.write(video_bbox_recode_filename)
            st.subheader("YOEO's Object Detection Results:")
            st.video(video_bbox_recode_filename)

            st.subheader("Flora & Fauna Detected: ")
            col1, col2, col3 = st.columns(3)
            col1.metric("# Species Detected", "2")
            col2.metric("Turtle", "1")
            col3.metric("Fish", "23")

        with tab_trim:
            st.subheader("YOEO's Trimmed Video:")

        with tab_beauty:
            st.subheader("YOEO's Beautiful Photos:")

        # remove recoded video to save space as it is not needed anymore
        os.remove(video_bbox_recode_filename)
        st.write(os.listdir(temp_path))

with st.expander("About YOEO"):
    st.write(
        "YOEO (You Only Edit Once) is an object detection model and web application created by data scientists and AI practitioners who are diving enthusiasts!"
    )
    st.write("The Model is trained on ...")


##STEP 3
# st.write("# 3. YOEO working its magic: ")
# st.write("-> to insert model inference and stich algo in progress bar")
# my_bar = st.progress(0)

# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)


##STEP 4
# st.write("# 4. Objects of interest detected and trimmed video output: ")

# col1, col2, col3 = st.columns(3)
# col1.metric("# Species Detected", "2")
# col2.metric("Turtle", "1")
# col3.metric("Fish", "23")

# st.video(vid_file)
