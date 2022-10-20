import os
import time

import cv2
import torch

from data.data_augment import ValTransform
from data.datasets import YOEO_CLASSES
from exp import get_exp
from utils import postprocess, vis


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=YOEO_CLASSES,
    ):
        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.preproc = ValTransform(legacy=False)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            # if self.decoder is not None:
            #     outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=True,
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, (bboxes, cls, cls_conf, scores)


def create_exp(exp_file, model_name, conf, nms, input_size):
    exp = get_exp(exp_file, model_name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = (input_size[0], input_size[1])

    return exp


def load_model(exp, ckpt_file):

    model = exp.get_model()
    model.eval()
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])

    return model


def video_predict(video_file, out_path, model, exp, YOEO_CLASSES, ifps, verbose=False):

    predictor = Predictor(model, exp, YOEO_CLASSES)

    cap = cv2.VideoCapture(video_file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    origi_shape = (width, height)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if verbose:
        print(num_frames, "frames detected!")

    orig_frames = []
    bbox_class_score = []

    vid_writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    index = 0
    start_time = time.time()
    while True:
        success, img = cap.read()
        index += 1
        if success:
            orig_frames.append(img)

            if index % (fps / ifps) == 0:  # inference optimization
                frame_start_time = time.time()
                outputs, img_info = predictor.inference(img)
                result_frame, results = predictor.visual(
                    outputs[0], img_info, predictor.confthre
                )
                vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

                if verbose:
                    print(
                        "--- Frame inferred in %0.2f seconds ---"
                        % (time.time() - frame_start_time)
                    )

                bbox_class_score.append(results)

            else:
                vid_writer.write(img)

        elif index > num_frames:
            break

    print("--- Completed in %0.2f seconds ---" % (time.time() - start_time))

    return (
        bbox_class_score,
        orig_frames,
        origi_shape,
        fps,
    )
