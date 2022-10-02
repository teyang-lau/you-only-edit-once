# python extract_frames.py -v ../data/videos -o ../data/extracted_frames -pvf ../data/ignore_prev_vid.txt -i True

import os
import uuid
import cv2
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Parse command line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-v", "--video_path", type=str, help="Path to video files")
parser.add_argument("-o", "--output_path", type=str, help="Path to output folder")
parser.add_argument("-pvf", "--prev_vid_file", type=str, default=None, help="Path to previous videos text file ")
parser.add_argument("-i", "--ignore_prev_vid", type=bool, default=False, help="Whether to ignore previous videos already extracted")

args = vars(parser.parse_args())

def video2frames( video_file, output_path ):

    """Extract frames from a video file
    
    Args:
    video_file (str): path to the video
    output_path (str): path to output folder for storing extracted frames

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    vid = cv2.VideoCapture(video_file)
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    index = 0        
    while vid.isOpened():
        success, img = vid.read()
        if success:
            index += 1
            # extract every fps frame of the video
            if index % fps == 0:
                cv2.imwrite(output_path + '/' + str(uuid.uuid4()) + '_' + str(index) + '.jpg', img)
        else:
            break
    vid.release()
    return

def multiple_video2frames( video_path, output_path , ignore_prev_vid=False, prev_vid_file=None):

    """Extract frames from multple videos file
    
    Args:
    video_path (str): path to folder containing all videos
    output_path (str): path to output folder for storing extracted frames
    prev_vid_file (str): path to text file containing previously extracted video filenames
    ignore_prev_vid (bool): whether to ignore previous vidoes that have been already extracted

    """

    if ignore_prev_vid:
        file = open(prev_vid_file)
        text = file.readlines()
        prev_vids = {t.rstrip('\n') : True for t in text} 
        file.close()
        file = open(prev_vid_file, 'a+')

    list_videos = os.listdir(video_path)
    for video in list_videos:
        # skip video if extracted before
        if ignore_prev_vid and video in prev_vids:
                continue
        # read and extract frame
        video_file = video_path + '/' + video
        video2frames(video_file, output_path)
        # add video name to ignore_prev_vid file
        if ignore_prev_vid:
            file.write(video + '\n')
            
    return

if __name__ == "__main__":
    multiple_video2frames(
        args['video_path'], 
        args['output_path'], 
        args['prev_vid_file'], 
        args['ignore_prev_vid'],
    )
