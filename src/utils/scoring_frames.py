"""
Score frames utils
"""

import numpy as np
from scipy.stats import mode


def bbox_area(bb):
    """Calculate area of bounding box

    Args:
    bb (list): bounding box corner xy coordinates [x1, y1, x2, y2]

    Returns:
    area (float): porportion of frame area covered by bbox
    """

    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    return width * height


def all_bbox_unionall_prop(bboxes):

    """
    Calculate proportion of frame covered by all bboxes
    Not ideal as it is computing unionall instead of union,
    so overlapping areas are counted twice
    """

    areas = [bbox_area(bb) for bb in bboxes]
    return sum(areas)


def all_bbox_unionall_prop_torch(bboxes, origi_shape):

    bbox_width = bboxes[:, 2] - bboxes[:, 0]
    bbox_height = bboxes[:, 3] - bboxes[:, 1]
    bbox_area_prop = (bbox_width * bbox_height) / (origi_shape[0] * origi_shape[1])

    return bbox_area_prop.sum()


def scores_over_all_frames(bbox_class_score, marine_options, origi_shape):
    area_scores, count_scores, marine_mask = [], [], []
    for frame in bbox_class_score:
        if len(frame[0]) == 0:
            area_scores.append(0)
            count_scores.append(0)
            marine_mask.append(False)
        else:
            area_scores.append(
                all_bbox_unionall_prop_torch(frame[0], origi_shape).item()
            )
            count_scores.append(frame[1].size()[0])
            marine_mask.append(
                True
                if set(frame[1].numpy().astype("int")) & set(marine_options)
                else False
            )

    return np.array(area_scores), np.array(count_scores), np.array(marine_mask)


def check_frames(bbox_class_score, marine_options):

    """
    Mask frames that have none of marine class specified by user &
    count number of objects detected in each frame

    Args:
    bbox_class_score (list): list of tuples for each frame's (bbox location, class, score)
    marine_options (list): list of classes that user wants to see

    Returns:
    mask (list): list of bool indicating whether there is any of marine class in each frame
    num_obj (list): list of count of objects detected in each frame

    """
    mask, num_obj = [], []
    for bb in bbox_class_score:
        # mask.append(any(x in bb[1] for x in marine_options))
        mask.append(True if set(bb[1]) & set(marine_options) else False)
        # count objects
        num_obj.append(len(bb[0]))
    return np.array(mask), np.array(num_obj)


def average_every_fps(arr, fps):
    # average every n element, where n is fps
    if len(arr) % fps == 0:
        avg = np.mean(arr.reshape(-1, fps), axis=1)
    else:
        avg = np.concatenate(
            (
                np.mean(arr[: (len(arr) // fps) * fps].reshape(-1, fps), axis=1),
                np.array([np.mean(arr[-(len(arr) % fps) :])]),
            )
        )

    return avg


def filter_frames(
    orig_frames, class_mask, area_scores, count_scores, threshold, strictness, fps, ifps
):

    """
    Filter frames based on whether frame contains any of user-specified classes
    Then, filter frames based on score using a moving past window.
    If current frame is to be kept, set previous n frames to be kept as well for smoother transitions

    Args:
    orig_frames (list): list of numpy arrays of original frames
    user_mask (list): list of numpy arrays of original frames
    frame_scores (np.array): array of scores for each frame
    threshold (float): threshold for filtering out frames based on scores
    strictness (int): number of prior frames to keep if current frame is to be kept

    Returns:
    filtered_frames (np.array): filtered array of frames
    filtered_scores (np.array): filtered array of frame scores
    idx_filtered_frames(np.array): indices of filtered frames

    """

    # min-max scale
    area_scores = (area_scores - np.min(area_scores)) / (
        np.max(area_scores) - np.min(area_scores)
    )
    count_scores = (count_scores - np.min(count_scores)) / (
        np.max(count_scores) - np.min(count_scores)
    )
    sum_scores = area_scores + count_scores
    mask = sum_scores >= threshold
    # mask = np.logical_or(mask, class_mask)  # maybe not needed

    # input mask to same size as orig frame
    mask = np.repeat(mask, fps / ifps)

    adjusted_mask = mask.copy()
    # moving past window
    adjusted_mask[
        np.maximum(np.flatnonzero(adjusted_mask) - strictness, 0)[:, None]
        + np.arange(strictness)
    ] = True
    # pad mask to same shape due to cutoffs from incomplete fps
    padded_mask = np.full((len(orig_frames)), False)
    padded_mask[: len(adjusted_mask)] = adjusted_mask
    padded_mask

    filtered_frames = np.array(orig_frames)[padded_mask]
    filtered_scores = sum_scores[adjusted_mask]
    # idx_filtered_frames = adjusted_mask.nonzero()[0]

    return filtered_frames, filtered_scores


def filter_area_and_count(
    area_scores, count_scores, marine_mask, threshold, strictness, fps, ifps, num_frames
):

    """
    Filter frames based on whether frame contains any of user-specified classes
    Then, filter frames based on score using a moving past window.
    If current frame is to be kept, set previous n frames to be kept as well for smoother transitions

    Args:
    orig_frames (list): list of numpy arrays of original frames
    user_mask (list): list of numpy arrays of original frames
    frame_scores (np.array): array of scores for each frame
    threshold (float): threshold for filtering out frames based on scores
    strictness (int): number of prior frames to keep if current frame is to be kept

    Returns:
    filtered_frames (np.array): filtered array of frames
    filtered_scores (np.array): filtered array of frame scores
    idx_filtered_frames(np.array): indices of filtered frames

    """

    # min-max scale
    area_scores = (area_scores - np.min(area_scores)) / (
        np.max(area_scores) - np.min(area_scores)
    )
    count_scores = (count_scores - np.min(count_scores)) / (
        np.max(count_scores) - np.min(count_scores)
    )
    sum_scores = area_scores + count_scores
    mask = sum_scores >= threshold
    mask = np.logical_and(mask, marine_mask)

    # input mask and sum_scores to same size as orig frame
    mask = np.repeat(mask, fps / ifps)
    sum_scores = np.repeat(sum_scores, fps / ifps)

    adjusted_mask = mask.copy()
    # moving past window
    adjusted_mask[
        np.maximum(np.flatnonzero(adjusted_mask) - strictness, 0)[:, None]
        + np.arange(strictness)
    ] = True

    padded_mask = np.full(num_frames, False)
    padded_mask[: len(adjusted_mask)] = adjusted_mask
    padded_mask

    filtered_scores = sum_scores[adjusted_mask]
    filtered_idx = (adjusted_mask == True).nonzero()

    return filtered_scores, filtered_idx[0]


def filter_area():
    pass


# def filter_frames(orig_frames, class_mask, frame_scores, threshold, strictness):

#     """
#     Filter frames based on whether frame contains any of user-specified classes
#     Then, filter frames based on score using a moving past window.
#     If current frame is to be kept, set previous n frames to be kept as well for smoother transitions

#     Args:
#     orig_frames (list): list of numpy arrays of original frames
#     user_mask (list): list of numpy arrays of original frames
#     frame_scores (np.array): array of scores for each frame
#     threshold (float): threshold for filtering out frames based on scores
#     strictness (int): number of prior frames to keep if current frame is to be kept

#     Returns:
#     filtered_frames (np.array): filtered array of frames
#     filtered_scores (np.array): filtered array of frame scores
#     idx_filtered_frames(np.array): indices of filtered frames

#     """

#     mask = frame_scores >= threshold
#     mask = np.logical_or(mask, class_mask)  # maybe not needed
#     adjusted_mask = mask.copy()
#     # moving past window
#     adjusted_mask[
#         np.maximum(np.flatnonzero(adjusted_mask) - strictness, 0)[:, None]
#         + np.arange(strictness)
#     ] = True
#     filtered_frames = np.array(orig_frames)[adjusted_mask]
#     filtered_scores = frame_scores[adjusted_mask]
#     idx_filtered_frames = adjusted_mask.nonzero()[0]

#     return filtered_frames, filtered_scores, idx_filtered_frames
