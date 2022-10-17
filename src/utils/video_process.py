import cv2
import os


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
