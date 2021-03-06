import argparse

import cv2
import numpy as np
import time
# from time import time
from pathlib import Path
import tflite_runtime.interpreter as tflite
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
# from networktables import NetworkTablesInstance
import cv2
import collections
import json
import sys
from wpi_helpers import ConfigParser, WPINetworkTables, ModelConfigParser

# class PBTXTParser:
#     def __init__(self, path):
#         self.path = path
#         self.file = None

#     def parse(self):
#         with open(self.path, 'r') as f:
#             self.file = ''.join([i.replace('item', '') for i in f.readlines()])
#             blocks = []
#             obj = ""
#             for i in self.file:
#                 if i == '}':
#                     obj += i
#                     blocks.append(obj)
#                     obj = ""
#                 else:
#                     obj += i
#             self.file = blocks
#             label_map = []
#             for obj in self.file:
#                 obj = [i for i in obj.split('\n') if i]
#                 name = obj[2].split()[1][1:-1]
#                 label_map.append(name)
#             self.file = label_map

#     def get_labels(self):
#         return self.file


class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

    def scale(self, sx, sy):
        """Returns scaled bounding box."""
        return BBox(xmin=sx * self.xmin,
                    ymin=sy * self.ymin,
                    xmax=sx * self.xmax,
                    ymax=sy * self.ymax)


class Tester:
    def __init__(self, config_parser):
        print("Initializing TFLite runtime interpreter")
        try:
            model_path = "rapid-react.tflite"
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = "rapid-react.tflite"
            print(model_path)
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Raspberry Pi CPU"

        print(self.hardware_type)

        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        print(f"Model input shape {self.input_detail['shape']}")
        print(f"Model output shape {self.output_details[0]['shape']}")

        ## Read the model configuration file
        print("Loading network settings")
        default_config_file = 'rapid-react-config.json'
        configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())    
        self.model_config = ModelConfigParser(configPath)

        print(self.model_config.labelMap)
        print("Classes:", self.model_config.classes)
        print("Confidence Threshold:", self.model_config.confidence_threshold)

        # Start the camera
        print("Starting camera server")
        cs = CameraServer.getInstance()
        cam = cs.startAutomaticCapture()
        camera_config = config_parser.cameras[1]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        cam.setResolution(WIDTH, HEIGHT)
        self.camera = cs.getVideo()
        self.img = np.zeros(shape=(320, 512, 3), dtype=np.uint8)
        print(self.img.shape)
        self.mjpegServer = cs.putVideo("TFLite", WIDTH, HEIGHT)
        self.frames = 0

        # Connect to WPILib Network Tables
        print("Connecting to Network Tables")
        self.nt = WPINetworkTables(config_parser.team, self.hardware_type, self.model_config.labelMap)

    def run(self):
        print("Starting mainloop")
        fps = 0.0
        tic = time.time()
        while True:
            # Acquire image frame 
            ret, frame_cv2 = self.camera.grabFrame(self.img)
            if not ret:
                print("Image failed")
                continue

            # Resize to expected shape [1xHxWx3] and input image into the model 
            scale = self.set_input(frame_cv2)

            # run inference
            self.interpreter.invoke()

            # Get inference output
            boxes, class_ids, scores, x_scale, y_scale = self.get_output(scale)
            for i in range(len(boxes)):
                if scores[i] > .5:

                    class_id = class_ids[i]
                    if np.isnan(class_id):
                        continue

                    class_id = int(class_id)
                    if class_id not in range(len(self.model_config.classes)):
                        continue

                    # Draw bounding boxes and label frame
                    cls_name = self.model_config.labelMap.get(int(class_id))
                    frame_cv2 = self.label_frame(frame_cv2, cls_name, boxes[i], scores[i], x_scale,
                                                 y_scale)

            # Display stream to browser                                     
            self.mjpegServer.putFrame(frame_cv2)

            # Put data to Network Tables
            self.nt.put_data(boxes, scores, class_ids, fps)

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

    def label_frame(self, frame, object_name, box, score, x_scale, y_scale):
        ymin, xmin, ymax, xmax = box
        score = float(score)
        bbox = BBox(xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax).scale(x_scale, y_scale)

        height, width, channels = frame.shape
        # check bbox validity
        if not 0 <= ymin < ymax <= height or not 0 <= xmin < xmax <= width:
            return frame

        ymin, xmin, ymax, xmax = int(bbox.ymin), int(bbox.xmin), int(bbox.ymax), int(bbox.xmax)
        self.temp_entry.append({"label": object_name, "box": {"ymin": ymin, "xmin": xmin, "ymax": ymax, "xmax": xmax},
                                "confidence": score})

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

        # Draw label
        # Look up object name from "labels" array using class index
        label = '%s: %d%%' % (object_name, score * 100)  # Example: 'person: 72%'
        label_size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
        label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10), (xmin + label_size[0], label_ymin + base - 10),
                      (255, 255, 255), cv2.FILLED)
        # Draw label text
        cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return frame

    def input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.input_detail['shape']
        return width, height

    def set_input(self, frame):
        """Copies a resized and properly zero-padded image to the input tensor.
        Args:
          frame: image
        Returns:
          Actual resize ratio, which should be passed to `get_output` function.
        """
        width, height = self.input_size()
        h, w, _ = frame.shape
        # print(f"Frame shape {frame.shape}")

        input_image_tensor = cv2.resize(frame.astype('float32'), (height, width))
        new_img = np.reshape(input_image_tensor, (1, height, width, 3))

        self.interpreter.set_tensor(self.input_detail['index'], np.copy(new_img))
        return width / w, height / h

    # def output_tensor(self, i):
    #     """Returns output tensor view."""
    #     # print(f"index {i}")
    #     tensor = self.interpreter.get_tensor(self.output_details[i]['index'])
    #     return np.squeeze(tensor)

    def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = [416,416]):
        scores_max = tflite.math.reduce_max(scores, axis=-1)

        mask = scores_max >= score_threshold
        class_boxes = tflite.boolean_mask(box_xywh, mask)
        pred_conf = tflite.boolean_mask(scores, mask)
        class_boxes = tflite.reshape(class_boxes, [tflite.shape(scores)[0], -1, tflite.shape(class_boxes)[-1]])
        pred_conf = tflite.reshape(pred_conf, [tflite.shape(scores)[0], -1, tflite.shape(pred_conf)[-1]])

        box_xy, box_wh = tflite.split(class_boxes, (2, 2), axis=-1)

        input_shape = tflite.cast(input_shape, dtype=tflite.float32)

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]

        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tflite.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tflite.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)

    def get_output(self, scale):
        pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        boxes, scores = self.filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=[width, height])

        boxes, scores, class_ids, valid_detections = tflite.image.combined_non_max_suppression(
            boxes=tflite.reshape(boxes, (tflite.shape(boxes)[0], -1, 1, 4)),
            scores=tflite.reshape(
                scores, (tflite.shape(scores)[0], -1, tflite.shape(scores)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )

        # boxes = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
        # print(boxes)
        # boxes = self.output_tensor(0)
        # class_ids = self.output_tensor(1)
        # scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale


if __name__ == '__main__':
    config_file = "/boot/frc.json"
    config_parser = ConfigParser(config_file)
    tester = Tester(config_parser)
    tester.run()
