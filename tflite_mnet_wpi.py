import argparse

import cv2
import numpy as np
from time import time
import tflite_runtime.interpreter as tflite
# from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer
# from networktables import NetworkTablesInstance
import cv2
import collections
import json
import sys
from pathlib import Path
from wpi_helpers import ConfigParser, WPINetworkTables, ModelConfigParser

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", action='store_true',
        help="path to input test image [False]")
    ap.set_defaults(image=False)   
    ap.add_argument("-m", "--model", type=str, required=False,
        default='rapid-react-mnet',
        help="model file path")  
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections, IoU threshold")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
        help="threshold when applying non-maxima suppression")
    ap.add_argument("-s", "--use_cv2_camera", action='store_true',
        help="use the OpenCV camera")  
    ap.set_defaults(use_cv2_camera=False)         
    ap.add_argument("-f", "--frc_config", type=str, required=False,
        default='/boot/frc.json',
        help="FRC config file path")  

    args = ap.parse_args()    
    return args

def start_cameraServer(WIDTH, HEIGHT):
    # pass
    from cscore import CameraServer
    print("Starting camera server")
    cs = CameraServer.getInstance()
    cam = cs.startAutomaticCapture()
    cam.setResolution(WIDTH, HEIGHT)
    return cs

def start_camera(WIDTH, HEIGHT):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return camera   

class PBTXTParser:
    def __init__(self, path):
        self.path = path
        self.file = None

    def parse(self):
        with open(self.path, 'r') as f:
            self.file = ''.join([i.replace('item', '') for i in f.readlines()])
            blocks = []
            obj = ""
            for i in self.file:
                if i == '}':
                    obj += i
                    blocks.append(obj)
                    obj = ""
                else:
                    obj += i
            self.file = blocks
            label_map = []
            for obj in self.file:
                obj = [i for i in obj.split('\n') if i]
                name = obj[2].split()[1][1:-1]
                label_map.append(name)
            self.file = label_map

    def get_labels(self):
        return self.file


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
    def __init__(self, args, config_parser):
        print("Initializing TFLite runtime interpreter")
        try:
            model_path = f"{args.model}.tflite"
            print(f"Model file path {model_path}")
            self.interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            self.hardware_type = "Coral Edge TPU"
        except:
            print("Failed to create Interpreter with Coral, switching to unoptimized")
            model_path = f"{args.model}.tflite"
            print(f"Model file path {model_path}")
            self.interpreter = tflite.Interpreter(model_path)
            self.hardware_type = "Unoptimized"
     
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()

        print(f"Model input shape {self.input_detail['shape']}")
        print(f"Model output shape {self.output_details[0]}")

        print("Getting labels")
        pbtxt_file = f"{args.model}.pbtxt"
        # pbtxtPath = str((Path(__file__).parent / Path(pbtxt_file)).resolve().absolute())
        parser = PBTXTParser(pbtxt_file)
        parser.parse()
        self.labels = parser.get_labels()

        print("Connecting to Network Tables")
        # ntinst = NetworkTablesInstance.getDefault()
        # ntinst.startClientTeam(config_parser.team)
        # ntinst.startDSClient()
        # self.entry = ntinst.getTable("ML").getEntry("detections")

        # self.coral_entry = ntinst.getTable("ML").getEntry("coral")
        # self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        # self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        # self.temp_entry = []

        print("Starting camera server")
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        if args.use_cv2_camera is True:
            # Use regular OpenCV camera
            self.camera = start_camera(WIDTH, HEIGHT)
            self.mjpegServer = False       
        else:    
            # Use robotpy-cscore camera server
            cs = start_cameraServer(WIDTH, HEIGHT)   
            self.camera = cs.getVideo()
            self.mjpegServer = cs.putVideo("OpenCV DNN", WIDTH, HEIGHT)

        # cs = CameraServer.getInstance()
        # camera = cs.startAutomaticCapture()
        # camera_config = config_parser.cameras[0]
        # WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
        # camera.setResolution(WIDTH, HEIGHT)
        # self.cvSink = cs.getVideo()
        # self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
        # self.output = cs.putVideo("Axon", WIDTH, HEIGHT)
        # self.frames = 0

        # self.coral_entry.setString(self.hardware_type)
        # self.resolution_entry.setString(str(WIDTH) + ", " + str(HEIGHT))

        # Connect to WPILib Network Tables
        print("Connecting to Network Tables")
        self.nt = WPINetworkTables(config_parser.team, self.hardware_type, self.labels)


    def run(self):
        print("Starting mainloop")
        while True:
            start = time()
            # Acquire frame and resize to expected shape [1xHxWx3]
            # ret, frame_cv2 = self.cvSink.grabFrame(self.img)
            # if not ret:
            #     print("Image failed")
            #     continue
            if args.use_cv2_camera is True:   
                # read from the OpenCV camera
                success, frame_cv2 = self.camera.read()
                if not success:
                    sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')            
            else:
                # read from robotpy-cscore camera server
                success, frame_cv2 = self.camera.grabFrame(self.img)   
                if not success:
                    print("Image failed")
                    continue    

            # input
            scale = self.set_input(frame_cv2)

            # run inference
            self.interpreter.invoke()

            # output
            boxes, class_ids, scores, x_scale, y_scale = self.get_output(scale)
            for i in range(len(boxes)):
                if scores[i] > .5:

                    class_id = class_ids[i]
                    if np.isnan(class_id):
                        continue

                    class_id = int(class_id)
                    if class_id not in range(len(self.labels)):
                        continue

                    frame_cv2 = self.label_frame(frame_cv2, self.labels[class_id], boxes[i], scores[i], x_scale,
                                                 y_scale)

            # self.output.putFrame(frame_cv2)

            # show the output image
            if self.mjpegServer is False:
                cv2.imshow("Image", frame_cv2)
            else:
                self.mjpegServer.putFrame(frame_cv2)

            # self.entry.setString(json.dumps(self.temp_entry))
            # self.temp_entry = []
            # if self.frames % 100 == 0:
            #     print("Completed", self.frames, "frames. FPS:", (1 / (time() - start)))
            # if self.frames % 10 == 0:
            #     self.fps_entry.setNumber((1 / (time() - start)))
            # self.frames += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
        self.temp_entry = []
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
        
        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = self.output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else: # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2

        h, w, _ = frame.shape
        floating_model = (self.input_detail['dtype'] == np.float32)
        if floating_model:
            new_img = np.reshape(cv2.resize(frame.astype('float32'), (320, 320)), (1, 320, 320, 3))
        else:
            new_img = np.reshape(cv2.resize(frame.astype('uint8'), (320, 320)), (1, 320, 320, 3))
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], np.copy(new_img))
        return width / w, height / h

    def output_tensor(self, i):
        """Returns output tensor view."""
        # print(f"output tensor {i}")
        tensor = self.interpreter.get_tensor(self.interpreter.get_output_details()[i]['index'])
        return np.squeeze(tensor)

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
        # pred = [self.interpreter.get_tensor(self.output_details[i]['index']) for i in range(len(self.output_details))]
        # boxes, scores = self.filter_boxes(pred[0], pred[1], score_threshold=0.25,
        #                                         input_shape=[width, height])

        # boxes, scores, class_ids, valid_detections = tflite.image.combined_non_max_suppression(
        #     boxes=tflite.reshape(boxes, (tflite.shape(boxes)[0], -1, 1, 4)),
        #     scores=tflite.reshape(
        #         scores, (tflite.shape(scores)[0], -1, tflite.shape(scores)[-1])),
        #     max_output_size_per_class=50,
        #     max_total_size=50,
        #     iou_threshold=0.45,
        #     score_threshold=0.25
        # )

        boxes = self.output_tensor(0)
        class_ids = self.output_tensor(1)
        scores = self.output_tensor(2)

        width, height = self.input_size()
        image_scale_x, image_scale_y = scale
        x_scale, y_scale = width / image_scale_x, height / image_scale_y
        return boxes, class_ids, scores, x_scale, y_scale


if __name__ == '__main__':
    args = parse_args()

    # Load the FRC configuration file
    frcConfigPath = str((Path(__file__).parent / Path(args.frc_config)).resolve().absolute())
    print(f"FRC config file path {frcConfigPath}")
    config_parser = ConfigParser(frcConfigPath)

    tester = Tester(args, config_parser)
    tester.run()
