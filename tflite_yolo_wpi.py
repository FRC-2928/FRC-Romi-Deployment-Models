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
            model_path = "model.tflite"
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = "rapid-react.tflite"
            print(model_path)
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Unoptimized"
            print(self.hardware_type)

        self.interpreter.allocate_tensors()

        ## Read the model configuration file
        print("Loading network settings")
        default_config_file = 'rapid-react-config.json'
        configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())    
        self.model_config = ModelConfigParser(configPath)
        print(self.model_config.labelMap)
        print("Classes:", self.model_config.classes)
        print("Confidence Threshold:", self.model_config.confidence_threshold)

        # print("Getting labels")
        # parser = PBTXTParser("map.pbtxt")
        # parser.parse()
        # self.labels = parser.get_labels()

        
        # print("Connecting to Network Tables")
        # ntinst = NetworkTablesInstance.getDefault()
        # ntinst.startClientTeam(config_parser.team)
        # ntinst.startDSClient()
        # self.entry = ntinst.getTable("ML").getEntry("detections")

        # self.coral_entry = ntinst.getTable("ML").getEntry("coral")
        # self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        # self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        # self.temp_entry = []

        print("Starting camera server")
        cs = CameraServer.getInstance()
        camera = cs.startAutomaticCapture()
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        camera.setResolution(WIDTH, HEIGHT)
        self.cvSink = cs.getVideo()
        self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.output = cs.putVideo("Axon", WIDTH, HEIGHT)
        self.frames = 0

        # Connect to WPILib Network Tables
        print("Connecting to Network Tables")
        hardware_type = "USB Camera"
        self.nt = WPINetworkTables(config_parser.team, hardware_type, self.model_config.labelMap)

        # self.hardware_entry.setString(self.hardware_type)
        # self.resolution_entry.setString(str(WIDTH) + ", " + str(HEIGHT))

    def run(self):
        print("Starting mainloop")
        fps = 0.0
        tic = time.time()
        while True:
            # Acquire frame and resize to expected shape [1xHxWx3]
            ret, frame_cv2 = self.cvSink.grabFrame(self.img)
            if not ret:
                print("Image failed")
                continue

            # input
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
            self.output.putFrame(frame_cv2)

            # Put data to Network Tables
            self.nt.put_data(boxes, scores, class_ids, fps)

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc

            # Put data to Network Tables
            # self.entry.setString(json.dumps(self.temp_entry))
            # self.temp_entry = []
            # if self.frames % 100 == 0:
            #     print("Completed", self.frames, "frames. FPS:", (1 / (time() - start)))
            # if self.frames % 10 == 0:
            #     self.fps_entry.setNumber((1 / (time() - start)))
            # self.frames += 1

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
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
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
        # new_img = np.reshape(cv2.resize(frame, (300, 300)), (1, 300, 300, 3))
        new_img = np.reshape(cv2.resize(frame, (240, 320)), (1, 320, 512, 3))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], np.copy(new_img))
        return width / w, height / h

    def output_tensor(self, i):
        """Returns output tensor view."""
        tensor = self.interpreter.get_tensor(self.interpreter.get_output_details()[i]['index'])
        return np.squeeze(tensor)

    def get_output(self, scale):
        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale


if __name__ == '__main__':
    config_file = "/boot/frc.json"
    config_parser = ConfigParser(config_file)
    tester = Tester(config_parser)
    tester.run()
