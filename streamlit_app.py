import streamlit as st
import cv2
import os
import tempfile
import matplotlib.pyplot as plt
from src.utils.streamlit import save_uploaded_file, factors
from yolox.yolox_process import create_exp, load_model, video_predict
from yolox.data.datasets import YOEO_CLASSES


MODEL_PATH = "./results/models/yolox_dive.pth"
MODEL_NAME = "yolox-nano"
EXP_PATH = "./exps/example/custom/nano.py"
MODEL_INPUT_SIZE = (640, 640)  # width, height
NUM_CLASSES = 5
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

##STEP 1 Load Model
with st.spinner(text="Loading Model ... Please be patient!"):
    exp = create_exp(
        EXP_PATH, MODEL_NAME, CONF_THRESHOLD, NMS_THRESHOLD, MODEL_INPUT_SIZE
    )
    model = load_model(exp, MODEL_PATH)

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

    # start inferencing
    trim_bt = st.button("Start Auto-Trimming!")
    st.write(trim_bt)
    if trim_bt:
        with st.spinner(text="YOEO working its magic: IN PROGRESS ..."):
            bbox_class_score, orig_frames, origi_shape, fps = video_predict(
                video_path,
                video_bbox_filename,
                model,
                exp,
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
