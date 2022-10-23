"""
Beautify Images Utils
"""

import random
import operator

import heapq
import math
from scipy.interpolate import UnivariateSpline

import cv2
import pilgram
from PIL import Image, ImageStat
import numpy as np

from src.utils.image_process import (
    do_we_need_to_sharpen,
    sharpen_my_image,
    adjust_contrast_brightness,
)


def get_top_frames(scores, num, fps, dispersed=True):
    """
    Returns list of indexes for number frames with the highest scores as
    specified by the user.

    Users can define the 'dispersed' function if they wish to have num images
    taken from different parts of the video. In this instance, we randomly sample
    10% of the frames from the video and score these frames.

    Otherwise the function just returns the best num images from the frames scored.
    """
    if len(scores) <= 1000:
        dispersed = False

    if dispersed:

        tmp = []

        while True:
            if len(tmp) == int(0.1 * len(scores)):
                break

            sampled_frame = random.choice(scores)

            if len(tmp) == 0:
                tmp.append(sampled_frame)
            else:
                flag = False

                for i in tmp:
                    if i - fps <= sampled_frame <= i + fps:
                        flag = True
                    break

                if flag == False:
                    tmp.append(sampled_frame)

        idx = sorted(
            list(zip(*heapq.nlargest(num, enumerate(tmp), key=operator.itemgetter(1))))[
                0
            ]
        )

        return sorted([scores.index(j) for j in [tmp[i] for i in idx]])

    else:
        return sorted(
            list(
                zip(*heapq.nlargest(num, enumerate(scores), key=operator.itemgetter(1)))
            )[0]
        )


def get_top_n_idx(filtered_scores, filtered_idx, sampling_size=0.1, n=10):
    """
    Random sample from scores and get the indices of the top n scores
    from original video

    Args:
    filtered_scores (np.array): scores filtered from object detection that pass a threshold
    filtered_idx (np.array): the indices of scores that pass the threshold, from original video
    sampling_size (float): proportion of samples to choose from num_frames of original video
    n (int): top n scores to choose from

    Return:
    top_n_idx (np.array): indices of top n scores from the sample,
    corresponding to indices from original video
    """

    # sample from filtered_scores & filtered_idx arrays
    n_sample = int(np.ceil(len(filtered_scores) * sampling_size))
    if n_sample <= n:
        n_sample = len(filtered_scores)
    rand_sample = np.random.choice(len(filtered_scores), n_sample, replace=False)
    rand_sample_scores = filtered_scores[rand_sample]
    rand_sample_idx = filtered_idx[rand_sample]

    # get the indices of the top n scores from the sample
    top_n_idx = rand_sample_idx[rand_sample_scores.argsort()[::-1][: min(n, n_sample)]]

    return top_n_idx


def brightness(im_file):
    """
    Returns perceived brightness of image
    https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    """
    stat = ImageStat.Stat(im_file)
    r, g, b = stat.mean
    return math.sqrt(0.241 * (r**2) + 0.691 * (g**2) + 0.068 * (b**2))


def LookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum = cv2.merge((blue_channel, green_channel, red_channel))
    return sum


def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel, red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win = cv2.merge((blue_channel, green_channel, red_channel))
    return win


def beautify(beauti_img, filter="hudson"):
    """
    Beautifies selected images.
    Input arguments:
    1) beauti_img (np.array) - array of images in cv2/BGR format
    2) filter (str) - instagram filter to apply
    3) The default filter is Hudson.
    List of Instagram filters: https://github.com/akiomik/pilgram/tree/master/pilgram
    """
    if filter:
        try:
            pilgram_filter = getattr(pilgram, filter.lower())

        except:
            raise ValueError(
                """
        That was not a correct filter. The list of correct filters are:
        _1977
        aden
        brannan
        brooklyn
        clarendon
        earlybird
        gingham
        hudson
        inkwell
        kelvin
        lark
        lofi
        maven
        mayfair
        moon
        nashville
        perpetua
        reyes
        rise
        slumber
        stinson
        toaster
        valencia
        walden
        willow
        xpro2
        Here's some showcases of filtered images:
        https://github.com/akiomik/pilgram/blob/master/screenshots/screenshot.png
        """
            )

    for idx, img in enumerate(beauti_img):

        ## Reduce blue light ##
        if filter and filter.lower() == "hudson":
            img = Summer(img)

        # Adjust brightness and contrast
        lux = brightness(Image.fromarray(img))
        if lux <= 130:
            beta = 137.5 - lux
        elif lux > 145:
            beta = 137.5 - lux
        else:
            beta = 0
        img = adjust_contrast_brightness(img, contrast=1.2, brightness=beta)

        ## Check and sharpen ##
        if do_we_need_to_sharpen(img):
            img = sharpen_my_image(img)

        ## Apply instagram filter ##
        if filter:
            img = np.array(pilgram_filter(Image.fromarray(img)))

        beauti_img[idx] = img

    return beauti_img


def check_filter(filter):

    error_msg = """
    That was not a correct filter. The list of correct filters are: \n
    _1977
    aden
    brannan
    brooklyn
    clarendon
    earlybird
    gingham
    hudson
    inkwell
    kelvin
    lark
    lofi
    maven
    mayfair
    moon
    nashville
    perpetua
    reyes
    rise
    slumber
    stinson
    toaster
    valencia
    walden
    willow
    xpro2
    \nHere's some showcases of filtered images:
    https://github.com/akiomik/pilgram/blob/master/screenshots/screenshot.png
    """

    try:
        pilgram_filter = getattr(pilgram, filter.lower())
    except:
        return False, error_msg

    return True, error_msg
