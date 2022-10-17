from __future__ import unicode_literals
import os
import cv2
import youtube_dl
import librosa
import ffmpeg
from pydub import AudioSegment

# import soundfile as sf
# import pafy
# from datetime import datetime


def video_stitch(
    images_array, video_out_path, video_name, origi_shape, fps, RGB2BGR=True
):

    """Stitch videos from images

    Args:
    images_array (list): list of numpy arrays of images
    video_out_path (str): path to output folder for storing stitched video
    video_name (onnx session): onnx session
    origi_shape (tuple): original shape of iamges/videos
    fps (int): intended frames per second of stitched video
    RGB2BGR (bool): whether to convert RGB2BGR for the images before stitching

    """

    out_vid_bbox = cv2.VideoWriter(
        os.path.join(video_out_path, video_name) + ".mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (origi_shape[1], origi_shape[0]),
    )
    for img_array in images_array:
        if RGB2BGR:
            out_vid_bbox.write(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        else:
            out_vid_bbox.write(img_array)
    out_vid_bbox.release()


def moviemaker(audio_file, video_file, output_path, youtube_url=False):
    """
    Creates a movie depending with the duration equivalent to the shortest duration
    between audio and video files.

    audio_file: Path to the audio file. This is required if youtube_url = False.
    If youtube_url = True, this can be set to None
    video_file: Path to the .mp4 file. This is required.
    output_path: This is required.
    youtube_url: The default is set to False
    """
    # Download audio from YouTube
    if youtube_url:
        audio_file = "./downloaded_audio.wav"

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_file,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "postprocessor_args": ["-ar", "16000"],
            "prefer_ffmpeg": True,
            "keepvideo": False,
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        print("Audio downloaded!")

    # Assemble video
    sound = AudioSegment.from_file(audio_file)
    audio_duration = librosa.get_duration(filename=audio_file)

    vid = cv2.VideoCapture(video_file)
    frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid.get(cv2.CAP_PROP_FPS)
    video_duration = frames / fps

    if audio_duration > video_duration:
        audio_file = "./audio_truncated.wav"
        first_cut_point = (0) * 1000  # miliseconds
        last_cut_point = round(video_duration) * 1000
        sound_clip = sound[first_cut_point:last_cut_point]
        sound_clip.export(audio_file, format="wav")

    else:
        clone = video_duration // audio_duration + 1
        audio_file = "./audio_extended.wav"
        sound *= clone  # the audio file is now longer than the video file
        first_cut_point = (0) * 1000
        last_cut_point = round(video_duration) * 1000
        sound_clip = sound[first_cut_point:last_cut_point]
        sound_clip.export(audio_file, format="wav")

    input_video = ffmpeg.input(video_file)
    input_audio = ffmpeg.input(audio_file)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path).run()

    print("Movie made! Congratulations!")
