"""
Onnx utils
"""

import onnx
import onnxruntime as ort
import time
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import time


# simple function to load a single image
def load_image_into_numpy_array(path, height, width):
    """
    Load an image from file into a numpy array.

    Args:
      path: the file path to the image
      height: height of image
      width: width of image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3), (original_height, original_width)
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((width, height))
    return np.array(image_resized), (image_shape[0], image_shape[1])


def convert_cv_image(cv_img, height, width):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_shape = img.shape

    img_resized = cv2.resize(img, (width, height))

    return np.array(img_resized), (img_shape[0], img_shape[1])


# simple function to read the label file label.pbtxt
def load_label_map(label_map_path):
    """
    Reads label map in the format of .pbtxt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    # creates the empty dictionary
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {
                    "id": label_index,
                    "name": label_name,
                }
    # outputs the dictionary of the label values
    return label_map


def post_process(
    rows_8400, num_classes, conf_thre=0.0001, nms_thre=0.3, print_info=False
):
    """
    Process onnx output and apply filtering using confidence & NMS threshold

    """

    # makes a copy of the input
    prediction = np.copy(rows_8400)
    # convert the copy to a tensor
    prediction = torch.Tensor(prediction)

    if print_info == True:
        print("Here is the input shape:", prediction.shape, "\n")
        print("Here is the input data:", "\n", prediction, "\n")

    # make a placeholder tensor of normal distribution/random values
    # we use this tensor to hold our calculated box positions temporarily
    box_corner = prediction.new(prediction.shape)
    if print_info == True:
        print("here is the created placeholder tensor:", "\n", box_corner, "\n")

    # perform some column maths magic to get the values for boxes location
    # col 0 - col 2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # col 1 - col 3
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # col 0 + col 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # col 1 + col 3
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # replace with new format
    prediction[:, :, :4] = box_corner[:, :, :4]
    # show the new prediction
    if print_info == True:
        print(
            "here is the updated 8400 rows of tensor with adjusted box coordinates:",
            "\n",
            prediction,
            "\n",
        )

    # trail and error to find the right column to start
    x_fac = 6

    # print the class probabilities
    if print_info == True:
        print(
            "here are the class probabilities:",
            "\n",
            prediction[:, :, x_fac : x_fac + num_classes],
            "\n",
        )

    # need to adjust the dimension based on the number of input classes
    if num_classes > 1:
        dims = 2
    else:
        dims = 0
    # print(dims)
    # get the class confidence and the index of the predicted class
    class_conf, class_pred = torch.max(
        prediction[:, :, x_fac : x_fac + num_classes], dims, keepdim=True
    )

    # get the masking for the rows above conf_thre
    conf_mask = (prediction[:, :, 4] * class_conf.squeeze() >= conf_thre).squeeze()

    # output the numebr of rows that are >= conf_thre
    rows_picked = 0
    for i in conf_mask:
        if i == True:
            rows_picked += 1

    if print_info == True:
        print("rows picked based on confidence threshold:", rows_picked, "\n")

    # concat all the newly prepared data into a new output tensor
    detections = torch.cat((prediction[:, :, :5], class_conf, class_pred.float()), 2)

    # apply masking to the 8400 rows, we only keep thos rows with > conf_thres
    detections = detections[:, conf_mask, :]
    if print_info == True:
        print("after masking:", "\n", detections)

    # apply the NMS to reduce the number of overlapping boxes
    d = detections[:, :, :4]
    d = d.view(len(detections[0]), 4)
    d.shape

    s = detections[:, :, 4] * detections[:, :, 5]
    s = s.view(len(detections[0]))
    s.shape

    id = detections[:, :, 6]
    id = id.view(len(detections[0]))
    id.shape

    # aglgorithm here
    nms_out_index = torchvision.ops.batched_nms(
        d,
        s,
        id,
        nms_thre,
    )
    if print_info == True:
        print("after NMS, rows pickedfor plotting:", "\n", nms_out_index, "\n")

    # apply the NMS masking
    output = detections[0][nms_out_index]

    # return the final output
    return output


def load_model(model_path, verbose=False):
    # test model
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Load model
    if verbose:
        print("Loading model...")
    start_time = time.time()
    session = ort.InferenceSession(model_path)
    if verbose:
        print("Model loaded, took {} seconds...".format(time.time() - start_time))

    # get the name of the input and output
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session, input_name, output_name


def pred_single_image(
    image_path,
    width,
    height,
    session,
    input_name,
    output_name,
    num_classes,
    conf_thre,
    nms_thre,
):
    # Returned resized + original picture in the format of width, height
    if isinstance(image_path, str):
        image_resized, origi_shape = load_image_into_numpy_array(
            image_path, int(width), int(height)
        )
    else:
        image_resized, origi_shape = convert_cv_image(
            image_path, int(width), int(height)
        )

    # preprocess the resized image so we can input them proper
    input_image = np.expand_dims(image_resized.astype(np.float32).transpose(2, 0, 1), 0)

    ## Feed image into model
    ort_outs = session.run([output_name], {input_name: input_image})

    # process onnx output and apply filtering using confidence & NMS threshold
    output = post_process(
        ort_outs[0],
        num_classes,
        conf_thre=conf_thre,
        nms_thre=nms_thre,
        print_info=False,
    )
    if output.shape[0] == 0:
        print("no bounding boxes were identified !")

    return output, image_resized, origi_shape


def get_bbox_class_score(onnx_output, width, height):

    # get the plotting information from the post processed outputs
    bboxes = onnx_output[:, 0:4].cpu().detach().numpy().astype(np.float64)
    classes = onnx_output[:, 6].cpu().detach().numpy().astype(np.float64)
    scores = (
        (onnx_output[:, 4] * onnx_output[:, 5])
        .cpu()
        .detach()
        .numpy()
        .astype(np.float64)
    )

    # perform some simple maths
    bboxes = [
        [
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ]
        for bbox in bboxes
    ]

    return bboxes, classes, scores


def draw_on_image(
    image_resized, origi_shape, bboxes, classes, scores, label_path, show_image=False
):

    # get the original image
    image_origi = Image.fromarray(image_resized).resize(
        (origi_shape[1], origi_shape[0])
    )
    # make into numpy array
    image_origi = np.array(image_origi)

    ## Load label map
    category_index = load_label_map(label_path)
    category_index

    np.random.seed(0)
    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [
            int(np.random.choice(range(256))),
            int(np.random.choice(range(256))),
            int(np.random.choice(range(256))),
            # int(i) for i in np.random.choice(range(256), size=3)
        ]

    # plot all the boxes for our current picture
    for idx, each_bbox in enumerate(bboxes):
        color = color_map.get(classes[idx])
        # print('plotted box',idx)
        # print('color:',color)

        ## Draw bounding box
        cv2.rectangle(
            image_origi,
            (
                int(each_bbox[1] * origi_shape[1]),
                int(each_bbox[0] * origi_shape[0]),
            ),
            (
                int(each_bbox[3] * origi_shape[1]),
                int(each_bbox[2] * origi_shape[0]),
            ),
            color,
            2,
        )

        ## Draw label background
        cv2.rectangle(
            image_origi,
            (
                int(each_bbox[1] * origi_shape[1]),
                int(each_bbox[2] * origi_shape[0]),
            ),
            (
                int(each_bbox[3] * origi_shape[1]),
                int(each_bbox[2] * origi_shape[0] + 15),
            ),
            color,
            -1,
        )

        ## Insert label class & score
        cv2.putText(
            image_origi,
            "Class: {}, Score: {}".format(
                str(category_index[classes[idx] + 1]["name"]),
                str(round(scores[idx], 2)),
            ),
            (
                int(each_bbox[1] * origi_shape[1]),
                int(each_bbox[2] * origi_shape[0] + 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    image_predict = Image.fromarray(image_origi)
    # image_predict
    if show_image:
        plt.imshow(image_predict)

    return image_predict


def video_predict(
    video_file,
    output_path,
    session,
    input_name,
    output_name,
    label_path,
    input_size,
    num_classes,
    conf_thre,
    nms_thre,
    verbose=False,
):

    """Extract frames from a video file or youtube link

    Args:
    video_file (str): path to the video
    output_path (str): path to output folder for storing extracted frames
    session (onnx session): onnx session
    input_name (str): onnx input name
    output_name (str): onnx output name
    label_path (str): path to label mapping .pbtxt file
    input_size (tuple): onnx model image input size
    num_classes (int): number of classes to predict
    conf_thre (float): confidence threshold for filtering out bboxes
    nms_thre (float): non-max supression threshold for filtering out bboxes
    verbose (bool): whether to print inference related info

    Return:
    frame_predictions (list): list of numpy arrays of frames with bbox drawn
    bbox_class_score (list): list of tuples for each frame's (bbox location, class, score)
    orig_frames (list): list of numpy arrays of original frames
    origi_shape (tuple): original shape of frames/video
    fps (int): original fps of video
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    vid = cv2.VideoCapture(video_file)

    frame_predictions = []
    orig_frames = []
    bbox_class_score = []
    fps = round(vid.get(cv2.CAP_PROP_FPS))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(num_frames, "frames detected!")
    index = 0
    start_time = time.time()
    while vid.isOpened():
        success, img = vid.read()
        index += 1
        if success:
            # extract every fps frame of the video and save
            # cv2.imwrite(
            #     output_path + "/" + str(index) + ".jpg",
            #     img,
            # )
            orig_frames.append(img)

            frame_start_time = time.time()
            # predict on image
            onnx_output, image_resized, origi_shape = pred_single_image(
                img,
                input_size[0],
                input_size[1],
                session,
                input_name,
                output_name,
                num_classes,
                conf_thre,
                nms_thre,
            )
            bboxes, classes, scores = get_bbox_class_score(
                onnx_output, input_size[0], input_size[1]
            )
            image_predict = draw_on_image(
                image_resized,
                origi_shape,
                bboxes,
                classes,
                scores,
                label_path,
                show_image=False,
            )
            bbox_class_score.append((bboxes, classes, scores))
            frame_predictions.append(np.array(image_predict))
            # save image_predict?
            if verbose:
                print(
                    "--- Frame inferred in %0.2f seconds ---"
                    % (time.time() - frame_start_time)
                )

        # stop reading at end of video
        # need this as some frames return False success, so cannot
        # use success to break the while loop
        if index > num_frames:
            break
    vid.release()

    print("--- Completed in %0.2f seconds ---" % (time.time() - start_time))

    return frame_predictions, bbox_class_score, orig_frames, origi_shape, fps
