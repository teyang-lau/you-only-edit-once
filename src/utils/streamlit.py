"""
Streamlit utils
"""

import os
import streamlit as st
from functools import reduce


def save_uploaded_file(uploadedfile, tempDir):
    orig_video_path = os.path.join(tempDir, "orig_video")
    if not os.path.exists(orig_video_path):
        os.makedirs(orig_video_path)
    with open(os.path.join(orig_video_path, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    st.success("Saved file :{} in tempDir".format(uploadedfile.name))

    return os.path.join(orig_video_path, uploadedfile.name)


def factors(n):
    return set(
        reduce(
            list.__add__,
            ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
        )
    )
