import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
import shutil
import gc
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils.streamlit import save_uploaded_file, factors, get_all_files, make_grid
from src.utils.yolox_process import video_predict
from src.utils.scoring_frames import scores_over_all_frames, filter_area_and_count
from src.utils.video_process import filter_video, add_audio
from src.utils.plotting import streamgraph
from src.utils.beautify import get_top_n_idx, beautify, check_filter
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead


@st.cache()
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

    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    return model


MODEL_PATH = "./results/models/yolox_dive.pth"
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
__, col, __ = st.columns([1, 2, 1])
with col:
    st.image("./results/media/you-only-edit-once-ai-logo.png")

##STEP 1 Load Model
with st.spinner(text="Loading Model ... Please be patient!"):
    model = load_model(MODEL_PATH)

with st.expander("How to Use YOEO"):
    instruct = """
    1. Upload a diving video 
    2. Select the flora & fauna you want to see in your video
    3. (Optional): Adjust advanced options for better fine-tuning and click "Submit Advanced Options"
    4. Click "Start Auto-Trimming" and let the magic begin! 
    """
    st.write(instruct)


# create temp dir for storing video and outputs
temp_dir = tempfile.TemporaryDirectory()
temp_path = temp_dir.name

##STEP 2 Upload Video
st.write("# Upload diving video:\n")

video_file = st.file_uploader(
    "Choose a File", accept_multiple_files=False, type=["mp4", "mov"]
)

if video_file is not None:
    file_details = {"FileName": video_file.name, "FileType": video_file.type}
    video_path = save_uploaded_file(video_file, temp_path)
    video_bbox_filename = os.path.join(temp_path, video_file.name)
    # get fps for optimization slider max value
    fps = round(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
    factors_fps = list(factors(fps))

    # user options
    st.markdown(
        """
        <style>
        .stMultiSelect > label {font-size:150%; font-weight:bold;}
        </style>
        """,
        unsafe_allow_html=True,
    )  # for all multi-select label sections
    marine_options = st.multiselect(
        "What flora & fauna do you prefer",
        ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
        ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
        help="Select the flora & fauna you want to be included in the final video",
    )
    marine_options_map = {
        "Shark": 0,
        "Corals": 1,
        "Fish": 2,
        "Turtle": 3,
        "Manta Ray": 4,
    }
    marine_options = [*map(marine_options_map.get, marine_options)]

    # user advanced options
    with st.expander("Advanced Options"):
        with st.form("my_form"):
            st.write("###### Leave as default if unsure!")
            ifps = st.select_slider(
                "Speed Optimization",
                options=factors_fps,
                value=max(factors_fps),
                help="Frames per sec to infer on. Smaller value means faster trimming but at the expense of performance!",
            )  # num of frames per sec to do inferencing
            strict_val = st.slider(
                "Trimming Strictness",
                min_value=0,
                max_value=int(fps * 3),
                value=round(fps / 2),
                help="Keep number of frames before an accepted frame to avoid objects from popping in instantly",
            )  # number of frames prior to keep if current frame is to be kept
            # sharpen = st.checkbox("Sharpen Video")
            # color_grade = st.checkbox("Color Grade Video")

            algo = st.radio(
                "Trimming Algorithm",
                ("Area", "Weighted Area & Count"),
                index=1,
                help="Algorithm for trimming the video based on object detection predictions.\
                    Area only uses the proportion of frame covered by bounding boxes.\
                    Weighted area & count additionally uses the counts of each detected species and\
                    weight them by a rarity score.",
            )
            if algo == "Weighted Area & Count":
                algo = "Area & Count"

            # audio = st.radio("Add Audio", ("No audio", "Default", "Youtube"), index=1)
            audio = st.radio(
                "Add Audio",
                ("No audio", "Default"),
                index=1,
                help="Whether to add audio to the trimmed video. Default will add a music chosen by us!",
            )
            # yt_link = st.text_input("Enter a Youtube Audio Link")

            filter = st.radio(
                "Add Filter",
                ("No filter", "Hudson", "Stinson", "Others"),
                index=2,
                help="Whether to add filter to beautified images.\
                    Choose Hudson for more saturation and Stinson for \
                    less saturation. Alternatively enter your choice of filter.\
                    Available Instagram filters: https://raw.githubusercontent.com/akiomik/pilgram/master/screenshots/screenshot.png",
            )
            other_filter = st.text_input("Other instagram filter")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit Advanced Options")
            # if submitted:
            #     st.write("Optimization", ifps, "strict_val", strict_val)

            # check other filter validity
            if other_filter and filter == "Others":
                check, error_msg = check_filter(other_filter)
                if not check:
                    st.error(error_msg)

    # start inferencing
    # background-color: #00cc00
    st.markdown(
        """<style>
        .row-widget.stButton:nth-of-type(1) button {
            color:black; font-weight: bold; font-size:20px; height:3em;
            width:15em; border-radius:10px 10px 10px 10px;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    trim_bt = st.button("Start Auto-Trimming!")
    if trim_bt:
        with st.spinner(
            text="YOEO working its magic: OBJECT DETECTION IN PROGRESS ..."
        ):
            bbox_class_score, origi_shape, fps, num_frames = video_predict(
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
            video_bbox_recode_filename = video_bbox_filename.replace(
                ".mp4", "_recoded.mp4"
            )
            video_bbox_recode_filename = video_bbox_recode_filename.replace(
                ".MP4", "_recoded.mp4"
            )
            os.system(
                "ffmpeg -i {} -vcodec libx264 {}".format(
                    os.path.join(video_bbox_filename),
                    video_bbox_recode_filename,
                )
            )

            # remove obj detect video to save space
            os.remove(video_bbox_filename)

            # score frames
            (
                area_scores,
                count_scores,
                marine_mask,
                species_detected,
                species_count,
                species_count_reorder,
            ) = scores_over_all_frames(bbox_class_score, marine_options, origi_shape)
            # filter scores for each frame
            filtered_scores, filtered_idx = filter_area_and_count(
                area_scores,
                species_count,
                marine_mask,
                1.1,
                strict_val,
                fps,
                ifps,
                num_frames,
                algo=algo,
            )
            # get top n indices of frames with highest scores
            beauti_idx = get_top_n_idx(
                filtered_scores, filtered_idx, sampling_size=0.1, n=10
            )
            # st.write("Proportion of filtered frames:", len(filtered_idx) / num_frames)
            # st.write("Number of filtered frames:", len(filtered_idx))

        with st.spinner(text="YOEO working its magic: Trimming ..."):
            # run through video and filter video
            video_trimmed_filename = os.path.join(
                temp_path, "orig_video", "trimmed.mp4"
            )
            video_trimmed_recode_filename = video_trimmed_filename.replace(
                ".mp4", "_recoded.mp4"
            )
            beauti_img = filter_video(
                video_path, video_trimmed_filename, filtered_idx, beauti_idx
            )  # beauti_img is now in order of appearance and not in terms of score

            # recode video and add audio
            audio_file = "./results/media/Retreat.mp3"
            if audio == "No audio":
                os.system(
                    "ffmpeg -i {}  -vcodec libx264 {}".format(
                        video_trimmed_filename, video_trimmed_recode_filename
                    )
                )
            # elif audio == "Youtube" and yt_link is not None:
            #     add_audio(
            #         audio_file,
            #         video_trimmed_filename,
            #         video_trimmed_recode_filename,
            #         yt_link,
            #     )
            else:  # default
                add_audio(
                    audio_file,
                    video_trimmed_filename,
                    video_trimmed_recode_filename,
                    temp_path,
                )
            # os.remove(video_trimmed_filename)

        # beautify images
        if other_filter and filter == "Others":
            # filter = other_filter
            beauti_img = beautify(beauti_img, filter=other_filter)
        elif filter == "No filter":
            # filter = None
            beauti_img = beautify(beauti_img, filter=None)
        else:
            beauti_img = beautify(beauti_img, filter=filter)

        tab_od, tab_trim, tab_beauty = st.tabs(
            [
                "YOEO's Object Detection Results",
                "Your Trimmed Video",
                "Beautiful Photos Captured By You",
            ]
        )
        with tab_od:
            st.subheader("YOEO's Object Detection Results:")
            st.video(video_bbox_recode_filename)

            # metrics
            st.subheader("Flora & Fauna Detected: ")
            st.markdown(
                """
                <style>
                /*center metric label*/
                [data-testid="stMetricLabel"] > div:nth-child(1) {
                    justify-content: center; font-size: 18px;
                }

                /*center metric value*/
                [data-testid="stMetricValue"] > div:nth-child(1) {
                    justify-content: center;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            __, col, col1, __ = st.columns([1, 3, 3, 1])
            col.metric("# Species Detected", len(species_detected))
            # bbox per frame
            col1.metric("Avg # Objects Per Frame", round(np.mean(count_scores)))
            col0, col1, col2, col3, col4 = st.columns(5)
            col0.metric("Fish", "Yes" if 2 in species_detected else None)
            col1.metric("Coral", "Yes" if 1 in species_detected else None)
            col2.metric("Turtle", "Yes" if 3 in species_detected else None)
            col3.metric("Shark", "Yes" if 0 in species_detected else None)
            col4.metric("Manta Ray", "Yes" if 4 in species_detected else None)

            # streamgraph plot
            with st.container():
                fig = streamgraph(species_count_reorder)
                st.pyplot(fig)

        with tab_trim:
            st.subheader("YOEO's Trimmed Video:")
            st.video(video_trimmed_recode_filename)

        with tab_beauty:
            st.subheader("Your Beautiful Photos:")
            row, col = 5, 2
            mygrid = make_grid(row, col)
            for idx, img in enumerate(beauti_img):
                mygrid[idx // col][idx % col].image(img, channels="BGR")

        # remove recoded video to save space as it is not needed anymore
        os.remove(video_bbox_recode_filename)
        os.remove(video_trimmed_recode_filename)

        # allfiles = get_all_files(temp_path)
        # st.write(allfiles)


with st.expander("About YOEO"):
    __, col2, __ = st.columns([1, 1, 1])
    with col2:
        st.image("./results/media/you-only-edit-once-ai-logo.png")

    about = """
    **[YOEO (You Only Edit Once)](https://github.com/teyang-lau/you-only-edit-once)** is an AI diving object detection model and diving 
    video trimming web application created by data scientists and 
    AI practitioners who are diving enthusiasts!
    
    **Created by:**
    * HE Xinyi
    * LAU TeYang
    * LI Zihao
    * LIM Hsien Yong
    * YAN Licheng
    """
    st.write(about)
    st.write("")
    model_text = """
    The Model is trained on ~500 fully annotated images from raw diving videos using
    [YOLOX-Nano](https://github.com/Megvii-BaseDetection/YOLOX) for 300 epochs.
    """
    st.markdown(model_text)
