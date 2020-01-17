import copy
import os
import time
import threading
from typing import List
import queue
import re

from absl import app, flags
from cv2 import cv2
from harvesters.core import Harvester
from harvesters.util.pfnc import (
    mono_location_formats,
    bayer_location_formats,
    rgb_formats,
    bgr_formats,
)
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# matplotlib.use("TKAgg")

MODEL_STATE_ROOT_DIR = "modelState"
MODEL_STATE_FILE_NAME = "modelState.pt"
DISPLAY_WINDOW_NAME = "Live Stream"
INFERENCE_WINDOW_NAME = "Inference"

exit_event = threading.Event()

flags.DEFINE_string(
    "gentl_producer_path",
    "/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti",
    "Path to the GenTL producer .cti file to use.",
)

flags.DEFINE_string(
    "local_data_dir", "../data", "Local directory of the image files to label."
)

flags.DEFINE_string(
    "label_file_path",
    "../data/labels.txt",
    "Path to the file containing the category labels.",
)

flags.DEFINE_float(
    "threshold", 0.5, "The threshold above which to display predicted bounding boxes"
)

flags.DEFINE_string("model_path", None, "The model to load. Default is newest.")


flags.DEFINE_integer("frame_rate", 30, "Frame rate to acquire images at.")


def draw_bboxes(
    ax, bboxes, label_indices, label_names, label_colors, label_scores=None
):
    for box_index, (box, label_index) in enumerate(zip(bboxes, label_indices)):
        height = box[3] - box[1]
        width = box[2] - box[0]
        lower_left = (box[0], box[1])
        rect = patches.Rectangle(
            lower_left,
            width,
            height,
            linewidth=2,
            edgecolor=label_colors[label_index],
            facecolor="none",
        )
        ax.add_patch(rect)
        label_string = ""
        if label_scores is None:
            label_string = label_names[label_index]
        else:
            label_string = "%s [%.2f]" % (
                label_names[label_index],
                label_scores[box_index],
            )
        ax.text(
            box[0],
            box[1] - 10,
            label_string,
            bbox=dict(
                facecolor=label_colors[label_index],
                alpha=0.5,
                pad=1,
                edgecolor=label_colors[label_index],
            ),
            fontsize=10,
            color="white",
        )


def int_string_sort(manifest_file) -> int:
    match = re.match("[0-9]+", manifest_file)
    if not match:
        return 0
    return int(match[0])


def get_newest_saved_model_path(model_dir_path: str) -> str:
    _, model_storage_dirs, _ = next(os.walk(model_dir_path))
    if len(model_storage_dirs) == 0:
        return None
    model_storage_dirs = sorted(model_storage_dirs, key=int_string_sort, reverse=True)
    model_file_path = os.path.join(
        model_dir_path, model_storage_dirs[0], MODEL_STATE_FILE_NAME
    )
    if not os.path.isfile(model_file_path):
        return None
    return model_file_path


def get_model_instance_detection(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class AcquiredImage:
    def __init__(
        self, width: int, height: int, data_format: str, image_data: np.ndarray
    ):
        self.width: int = width
        self.height: int = height
        self.data_format: str = data_format
        self.image_data: np.ndarray = image_data
        self.processed = False

    def get_data(self, process: bool = False) -> np.ndarray:
        if process:
            self.process_image()
        return self.image_data

    def process_image(self) -> None:
        if self.processed:
            return

        if self.data_format in mono_location_formats:
            self.image_data = self.image_data.reshape(self.height, self.width)
            self.processed = True
        elif self.data_format == "BayerRG8":
            self.image_data = cv2.cvtColor(
                self.image_data.reshape(self.height, self.width), cv2.COLOR_BayerRG2RGB
            )
            self.data_format == "RGB8"
            self.processed = True
        elif self.data_format in rgb_formats or self.data_format in bgr_formats:

            self.image_data = self.image_data.reshape(self.height, self.width, 3)

            if self.data_format in bgr_formats:
                # Swap every R and B:
                content = content[:, :, ::-1]
            self.processed = True
        else:
            print("Unsupported pixel format: %s" % self.data_format)

    def get_resized_image(self, target_width: int) -> np.ndarray:
        resize_ratio = float(target_width / self.width)
        return cv2.resize(self.image_data, (0, 0), fx=resize_ratio, fy=resize_ratio)

    def save(self, file_path: str) -> bool:
        try:
            cv2.imwrite(file_path, self.get_data())
        except:
            return False
        return True


def acquire_images(cam, consumer_queues: List[queue.Queue]) -> None:
    cam.start_image_acquisition()
    print("Acquisition started.")
    while not exit_event.is_set():
        with cam.fetch_buffer() as buffer:
            component = buffer.payload.components[0]
            for q in consumer_queues:

                # clear stale images
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass

                q.put(
                    AcquiredImage(
                        component.width,
                        component.height,
                        component.data_format,
                        component.data.copy(),
                    )
                )
    # tell consumer queues that acquisition has ended

    for q in consumer_queues:
        # clear stale images
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        q.put(None)

    cam.stop_image_acquisition()
    print("Acquisition Ended.")


def display_images(acquisition_queue: queue.Queue, display_mutex) -> None:
    try:
        with display_mutex:
            cv2.namedWindow(DISPLAY_WINDOW_NAME)
            cv2.moveWindow(DISPLAY_WINDOW_NAME, 0, 0)
        print("Starting live stream.")

        while True:

            retrieved_image = acquisition_queue.get(block=True)
            if retrieved_image is None:
                break

            retrieved_image.process_image()
            with display_mutex:
                # print("Show image")
                cv2.imshow(
                    DISPLAY_WINDOW_NAME,
                    retrieved_image.get_resized_image(target_width=720),
                )
                cv2.waitKey(1)

    finally:
        # end acquisition if there are any issues
        exit_event.set()
        print("Ending live stream")
        cv2.destroyWindow(DISPLAY_WINDOW_NAME)


def predict_images(
    inference_queue: queue.Queue, labels, saved_model_file_path, display_mutex
) -> None:
    try:
        with display_mutex:
            cv2.namedWindow(INFERENCE_WINDOW_NAME)
            cv2.moveWindow(INFERENCE_WINDOW_NAME, 1024, 0)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # get the model using our helper function
        model = get_model_instance_detection(len(labels))

        print("Loading model state from: %s" % saved_model_file_path)

        model.load_state_dict(torch.load(saved_model_file_path))

        # move model to the right device
        model.to(device)

        model.eval()

        print("Model state loaded")

        label_colors = plt.get_cmap("hsv")(np.linspace(0, 0.9, len(labels)))

        print("Starting inference")

        while True:

            retrieved_image = inference_queue.get(block=True)
            if retrieved_image is None:
                break

            retrieved_image.process_image()

            pil_image = F.to_tensor(
                Image.fromarray((retrieved_image.get_data() / 255.0), mode="RGB")
            )

            outputs = model([pil_image])
            outputs = [
                {k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs
            ]

            # with display_mutex:
            #     cv2.imshow(
            #         INFERENCE_WINDOW_NAME,
            #         retrieved_image.get_resized_image(target_width=720),
            #     )

            # filter out the background labels and scores bellow threshold
            filtered_output = [
                (
                    outputs[0]["boxes"][j],
                    outputs[0]["labels"][j],
                    outputs[0]["scores"][j],
                )
                for j in range(len(outputs[0]["boxes"]))
                if outputs[0]["scores"][j] > flags.FLAGS.threshold
                and outputs[0]["labels"][j] > 0
            ]

            inference_boxes, inference_labels, inference_scores = (
                zip(*filtered_output) if len(filtered_output) > 0 else ([], [], [])
            )

            # draw_bboxes(
            #     inference_ax,
            #     inference_boxes,
            #     inference_labels,
            #     labels,
            #     label_colors,
            #     inference_scores,
            # )

            # cv2.waitKey(1)

            # plt.pause(0.001)
            # if plt.fignum_exists(inference_ax.figure.number):
            #     inference_ax.figure.canvas.draw()

    finally:
        # end acquisition if there are any issues
        exit_event.set()
        print("Ending inference")
        # cv2.destroyWindow(INFERENCE_WINDOW_NAME)


def apply_camera_settings(cam) -> None:
    cam.remote_device.node_map.AcquisitionFrameRateEnable.value = True
    cam.remote_device.node_map.AcquisitionFrameRate.value = min(
        flags.FLAGS.frame_rate, cam.remote_device.node_map.AcquisitionFrameRate.max
    )
    print(
        "Acquisition frame rate set to: %3.1f"
        % cam.remote_device.node_map.AcquisitionFrameRate.value
    )


def main(unused_argv):

    if not os.path.isfile(flags.FLAGS.label_file_path):
        print("Invalid category labels path.")
        return

    labels = [
        label.strip() for label in open(flags.FLAGS.label_file_path).read().splitlines()
    ]

    if len(labels) == 0:
        print("No labels are present in %s" % flags.FLAGS.label_file_path)
        return

    # Add the background as the first class
    labels.insert(0, "background")

    print("Labels found:")
    print(labels)

    saved_model_file_path = (
        flags.FLAGS.model_path
        if flags.FLAGS.model_path is not None
        else get_newest_saved_model_path(
            os.path.join(flags.FLAGS.local_data_dir, MODEL_STATE_ROOT_DIR)
        )
    )

    if saved_model_file_path is None:
        print("No saved model state found")
        return

    h = Harvester()
    h.add_cti_file(flags.FLAGS.gentl_producer_path)
    if len(h.cti_files) == 0:
        print("No valid cti file found at %s" % flags.FLAGS.gentl_producer_path)
        h.reset()
        return
    print("Currently available genTL Producer CTI files: ", h.cti_files)

    h.update_device_info_list()
    if len(h.device_info_list) == 0:
        print("No compatible devices detected.")
        h.reset()
        return

    print("Available devices List: ", h.device_info_list)
    print("Using device: ", h.device_info_list[0])

    cam = h.create_image_acquirer(list_index=0)

    apply_camera_settings(cam)

    # Newest only single image queue
    acquisition_queue = queue.Queue(1)
    inference_queue = queue.Queue(1)

    display_mutex = threading.Lock()

    acquire_thread = threading.Thread(
        target=acquire_images, args=(cam, [acquisition_queue, inference_queue],)
    )
    display_thread = threading.Thread(
        target=display_images, args=(acquisition_queue, display_mutex,)
    )

    inference_thread = threading.Thread(
        target=predict_images,
        args=(inference_queue, labels, saved_model_file_path, display_mutex,),
    )

    inference_thread.start()
    display_thread.start()
    acquire_thread.start()

    # while 1:
    #     print("Refresh")
    # with display_mutex:
    # cv2.waitKey(1)
    #     if keypress == 27:
    #         # escape key pressed
    #         exit_event.set()
    # print("Exiting")

    inference_thread.join()
    display_thread.join()
    acquire_thread.join()

    # clean up
    cam.destroy()
    h.reset()

    print("Exiting.")


if __name__ == "__main__":
    app.run(main)
