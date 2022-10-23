import streamlit as st
import cv2
import time
import os
import tempfile
import matplotlib.pyplot as plt
from src.utils.onnx_process import load_model, video_predict
from src.utils.video_process import video_stitch
from src.utils.streamlit import save_uploaded_file

options = st.multiselect(
    "What flora & fauna do you prefer",
    ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
    ["Fish", "Coral", "Turtle", "Shark", "Manta Ray"],
    help="Select the flora & fauna you want to be included in the final video",
)
with st.expander("Advanced Options"):
    with st.form("my_form"):
        # st.write("Inside the form")
        factor_fps = [1, 2, 3, 5, 6, 10, 15, 30]
        op_val = st.select_slider("Optimization", options=factor_fps, value=30)
        strict_val = st.slider("Trimming Strictness")
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("optimization", op_val, "strictness", strict_val)

st.write("Outside the form")

st.snow()
