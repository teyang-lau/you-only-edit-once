"""
Beautify Images Utils
"""

import os
import random
import operator

import heapq
import math
from scipy.interpolate import UnivariateSpline

import cv2
import pilgram
from PIL import Image,ImageFilter, ImageStat
import numpy as np

from src.utils.image_process import *

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
  '''
  Returns perceived brightness of image
  https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
  '''
  stat = ImageStat.Stat(im_file)
  r,g,b = stat.mean
  return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

def LookupTable(x, y):
  spline = UnivariateSpline(x, y)
  return spline(range(256))

def Summer(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel  = cv2.split(img)
    red_channel = cv2.LUT(red_channel, increaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decreaseLookupTable).astype(np.uint8)
    sum= cv2.merge((blue_channel, green_channel, red_channel ))
    return sum

def Winter(img):
    increaseLookupTable = LookupTable([0, 64, 128, 256], [0, 80, 160, 256])
    decreaseLookupTable = LookupTable([0, 64, 128, 256], [0, 50, 100, 256])
    blue_channel, green_channel,red_channel = cv2.split(img)
    red_channel = cv2.LUT(red_channel, decreaseLookupTable).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increaseLookupTable).astype(np.uint8)
    win= cv2.merge((blue_channel, green_channel, red_channel))
    return win

def beautify(image_directory, indices, output_path, manual=False, filter = 'hudson'):
  '''
  Beautifies selected images.
  Input arguments:
  1) image_directory - directory where images are stored
  2) indices - indices for the images to edit
  3) output_path - path to export images
  4) manual - users can manually beautify their images using 
  instagram filters
  5) The default filter is Hudson.
  List of Instagram filters: https://github.com/akiomik/pilgram/tree/master/pilgram
  '''
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  imgs = [os.listdir(image_directory)[i] for i in indices]

  try:
    filter = getattr(pilgram, filter)
    
  except:
    print("""
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
    """)

  for img_name in imgs:
    input_path = os.path.join(image_directory, img_name)
    filename = img_name.split(".")[0]

    img_p = Image.open(input_path)
    lux = brightness(img_p)

    if not manual:
      img = cv2.imread(input_path)

      ## Check and sharpen ##
      if do_we_need_to_sharpen(img)==True:
        img = sharpen_my_image(img)
      
      ## Check and adjust brightness ##
      if lux <=90:
        img = cv2.convertScaleAbs(img, beta=95-lux)
        
      elif lux>100:
        img = cv2.convertScaleAbs(img, beta=95-lux)
        
      ## Reduce blue light ##
      img = Summer(img)
      cv2.imwrite(f'{output_path}/{filename}_enhanced.png', img)

      ## Icing on the cake ##
      img = Image.open(f'{output_path}/{filename}_enhanced.png')
      filter(img).save(f'{output_path}/{filename}_enhanced.png')

    else:
      filter(img).save(f'{output_path}/{filename}_enhanced.png')

  print('Image beautified! Enjoy!')
