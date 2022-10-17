# python extract_frames.py -v ../data/videos -o ../data/extracted_frames -pvf ../data/ignore_prev_vid.txt -i True

import os
import uuid
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
def parse_args():

    """Parse command line arguments."""

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--video_path", type=str, help="Path to video files")
    parser.add_argument("-o", "--output_path", type=str, help="Path to output folder")
    parser.add_argument(
        "-pvf",
        "--prev_vid_file",
        type=str,
        default=None,
        help="Path to previous videos text file ",
    )
    parser.add_argument(
        "-i",
        "--ignore_prev_vid",
        type=bool,
        default=False,
        help="Whether to ignore previous videos already extracted",
    )

    return vars(parser.parse_args())


def video2frames(video_file, output_path, factor=1, youtube=False):

    """Extract frames from a video file or youtube link

    Args:
    video_file (str): path to the video
    output_path (str): path to output folder for storing extracted frames
    factor (int): how many seconds to extract 1 frame. 1 = extract a frame every sec, 2 = extract a frame every 2 secs
    youtube (bool): whether to get video directly from youtube link

    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if youtube == False:
        vid = cv2.VideoCapture(video_file)

    elif youtube == True:
        video = pafy.new(video_file)
        best = video.getbest(preftype="mp4")
        vid = cv2.VideoCapture(best.url)

    fps = round(vid.get(cv2.CAP_PROP_FPS))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    index = 0
    while vid.isOpened():
        success, img = vid.read()
        index += 1
        if success:
            # extract every fps frame of the video, multplied by a factor
            # factor of 1 = extract a frame every sec, 2 = extract a frame every 2 secs
            if index % (fps * factor) == 0:
                cv2.imwrite(
                    output_path + "/" + str(uuid.uuid4()) + "_" + str(index) + ".jpg",
                    img,
                )
        # stop reading at end of video
        # need this as some frames return False success, so cannot
        # use success to break the while loop
        if index > num_frames:
            break
    vid.release()

    return


def multiple_video2frames(
    video_path, output_path, factor=1, ignore_prev_vid=False, prev_vid_file=None
):

    """Extract frames from multple videos file

    Args:
    video_path (str): path to folder containing all videos
    output_path (str): path to output folder for storing extracted frames
    ignore_prev_vid (bool): whether to ignore previous vidoes that have been already extracted
    prev_vid_file (str): path to text file containing previously extracted video filenames

    """

    vid_count = 0

    if ignore_prev_vid:
        file = open(prev_vid_file)
        text = file.readlines()
        prev_vids = {t.rstrip("\n"): True for t in text}
        file.close()

    list_videos = os.listdir(video_path)
    print("Found {} videos".format(len(list_videos)))
    for video in list_videos:
        # skip video if extracted before
        if ignore_prev_vid and video in prev_vids:
            continue
        # read and extract frame
        vid_count += 1
        print("Extracting Video {}".format(vid_count))
        video_file = video_path + "/" + video
        video2frames(video_file, output_path, factor=factor)
        # add video name to ignore_prev_vid file
        if ignore_prev_vid:
            file = open(prev_vid_file, "a+")
            file.write(video + "\n")
            file.close()

    if vid_count > 0:
        print("Extraction Completed!")

    return


def read_text_files(file_path):
    with open(file_path, "r") as f:
        print(f.read())

    return


def trytry(video_path, output_path, ignore_prev_vid=False, prev_vid_file=None):

    list_videos = os.listdir(video_path)
    print(list_videos)
    with open(prev_vid_file, "r") as f:
        print(f.read())


if __name__ == "__main__":
    args = parse_args()
    multiple_video2frames(
        args["video_path"],
        args["output_path"],
        args["prev_vid_file"],
        args["ignore_prev_vid"],
    )
