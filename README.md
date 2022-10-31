# <div align="center">You Only Edit Once</div>

<div align="center">

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://www.python.org/)
[![made-with-Markdown](https://img.shields.io/badge/Made%20with-Markdown-1f425f.svg)](http://commonmark.org)
[![Generic badge](https://img.shields.io/badge/STATUS-COMPLETED-<COLOR>.svg)](https://shields.io/)
[![GitHub license](https://img.shields.io/github/license/teyang-lau/HDB_Resale_Prices.svg)](https://github.com/teyang-lau/HDB_Resale_Prices/blob/main/LICENSE)
<br>
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://you-only-edit-once.streamlitapp.com/)
<a href="https://colab.research.google.com/github/teyang-lau/you-only-edit-once/blob/main/notebooks/postprocessing/you_only_edit_once.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

<p align="center">
  <img src="results\media\you-only-edit-once-ai-logo.png" width="300">
  <br><br>
  You Only Edit Once (YOEO) 🧠 is an AI-powered video editing tool 🎥 for beginner and leisure divers 🤿. It uses state-of-the-art deep learning object detection models to automatically trim 🎬 underwater diving videos and beautify selected images 📸!
</p>

<br>


## <div align="center">YOEO Web Application</div>
  
<div align="center">
  <p>
    Check out the <b><a href="https://you-only-edit-once.streamlitapp.com/">You Only Edit Once web application</a></b> hosted on streamlit!
  </p>
</div>

<br>

## <div align="center">Object Detection Output Example</div>
https://user-images.githubusercontent.com/58768271/198932663-c15927b5-c111-4062-b5f3-126c31df4124.mp4

<br>

## <div align="center">Final Trimmed Video Example (Turn on the audio!)</div>
https://user-images.githubusercontent.com/58768271/198932745-926518b2-44e5-4f17-b627-a99fb8cb62e3.mp4

<br>

## <div align="center">Project Directory Structure</div>

```
.
├── notebooks           <- notebooks for explorations / debugging / training
│   ├── preprocessing
│   ├── postprocessing
│   ├── training
├── results             <- storing results and outputs
│   ├── media           <- store multimedia data
│   ├── others          <- miscellaneous 
│   ├── models          <- Trained and serialized models, model predictions, or model summaries
├── src                 <- all source code, internal org as needed
│   ├── utils           <- utility functions
├── yolox               <- yolox modules
├── requirements.txt    <- installing dependencies   
└── streamlit_app.py    <- streamlit deployment script
```
