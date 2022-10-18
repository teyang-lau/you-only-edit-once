"""
Score frames utils
"""

import numpy as np


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


def filter_frames(orig_frames, frame_scores, area_threshold, strictness):
    mask = frame_scores >= area_threshold
    adjusted_mask = mask.copy()
    adjusted_mask[
        np.maximum(np.flatnonzero(adjusted_mask) - strictness, 0)[:, None]
        + np.arange(strictness)
    ] = True
    filtered_frames = np.array(orig_frames)[adjusted_mask]
    filtered_scores = frame_scores[adjusted_mask]

    # return filtered_scores, idx_filtered_scores
