#!/usr/bin/env python3

import os
import json
import time
from time import sleep
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from PIL import Image
from pathlib import Path
import sys
import numpy as np
import cv2
from networktables import NetworkTablesInstance

# Constants
FRAME_WIDTH = 416
FRAME_HEIGHT = 416

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

class ConfigParser:
    def __init__(self):
        self.team = -1

        # Get the FRC config file path. For WPILibPi Romi image this file 
        # is at /boot.  
        config_path = os.path.join('/boot', 'frc.json')

        # For testing check the current directory
        if not Path(config_path).exists():
            # Use the file with this package if not running on the Romi image
            config_path = os.path.join('frc.json')

        print("Using config", config_path)

        # parse file
        try:
            with open(config_path, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

        # top level must be an object
        if not isinstance(j, dict):
            self.parseError("must be JSON object", config_path)

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parseError("could not read team number", config_path)

        # cameras
        try:
            self.cameras = j["cameras"]
        except KeyError:
            self.parseError("could not read cameras", config_path)

    def parseError(self, str, config_file):
        """Report parse error."""
        print("config error in '" + config_file + "': " + str, file=sys.stderr)     

# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

class ModelConfigParser:
    def __init__(self, path):
        """
        Parses the model config file and adjusts NNetManager values accordingly. 
        It's advised to create a config file for every new network, as it allows to 
        use dedicated NN nodes (for `MobilenetSSD <https://github.com/luxonis/depthai/blob/main/resources/nn/mobilenet-ssd/mobilenet-ssd.json>`__ 
        and `YOLO <https://github.com/luxonis/depthai/blob/main/resources/nn/tiny-yolo-v3/tiny-yolo-v3.json>`__)
        or use `custom handler <https://github.com/luxonis/depthai/blob/main/resources/nn/openpose2/openpose2.json>`__ 
        to process and display custom network results

        Args:
            path (pathlib.Path): Path to model config file (.json)

        Raises:
            ValueError: If path to config file does not exist
            RuntimeError: If custom handler does not contain :code:`draw` or :code:`show` methods
        """
        configPath = Path(path)
        if not configPath.exists():
            raise ValueError("Path {} does not exist!".format(path))

        with configPath.open() as f:
            configJson = json.load(f)
            nnConfig = configJson.get("nn_config", {})
            labels = configJson.get("mappings", {}).get("labels", None)
            self.labelMap = {i: n for i, n in enumerate(labels)}
            self.nnFamily = nnConfig.get("NN_family", None)
            self.outputFormat = nnConfig.get("output_format", "raw")
            metadata = nnConfig.get("NN_specific_metadata", {})
            if "input_size" in nnConfig:
                self.inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))

            self.confidence_threshold = metadata.get("confidence_threshold", nnConfig.get("confidence_threshold", None))
            self.classes = metadata.get("classes", None)

class Camera():
    def __init__(self, config_parser):
        camera_config = config_parser.cameras[0]
        WIDTH, HEIGHT = camera_config["width"], camera_config["height"]

        try:
            # Use CameraServer if installed
            from cscore import CameraServer
            print("Starting camera server")
            cs = CameraServer.getInstance()
            cam = cs.startAutomaticCapture()
            cam.setResolution(WIDTH, HEIGHT)
            camera = cs.getVideo()
            self.img = np.zeros(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8)
            self.mjpegServer = cs.putVideo("OpenCV DNN", WIDTH, HEIGHT)
        except ModuleNotFoundError:
            # Else use OpenCV camera
            print("Starting cv2 camera")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            self.mjpegServer = False

        self.camera = camera   

    def read(self):
        if self.mjpegServer is False:
            return self.camera.read()   
        else:
            return self.camera.grabFrame(self.img)

    def output(self, frame):    
        # show the output image
        if self.mjpegServer is False:
            cv2.imshow("Image", frame)
        else:
            self.mjpegServer.putFrame(frame)          
        

class WPINetworkTables():
    """
        The WPINetworkTables class is used to send inference data back to the WPI program.

    # Arguments
      labelMap: a dictionary used to translate class id to its name.
    """

    def __init__(self, team, hardware_type, labelMap):
        self.labelMap = labelMap

        ntinst = NetworkTablesInstance.getDefault()
        ntinst.startClientTeam(team)
        ntinst.startDSClient()
        
        self.hardware_entry = ntinst.getTable("ML").getEntry("device")
        self.fps_entry = ntinst.getTable("ML").getEntry("fps")
        self.resolution_entry = ntinst.getTable("ML").getEntry("resolution")
        self.entry = ntinst.getTable("ML").getEntry("detections")

        self.hardware_entry.setString(hardware_type)
        self.resolution_entry.setString(str(FRAME_WIDTH) + ", " + str(FRAME_HEIGHT)) 
        self.startTime = time.monotonic()
        self.fps = 0

    def put_data(self, boxes, confidence, class_ids):
        
        for bb, cf, cl in zip(boxes, confidence, class_ids):
            temp_entry = []
            cls_name = self.labelMap[int(cl)]
            xmin, ymin, xmax, ymax = bb[0], bb[1], bb[2], bb[3]
            temp_entry.append({"label": cls_name, 
                                "box": {"ymin": int(ymin), "xmin": int(xmin), "ymax": int(ymax), "xmax": int(xmax)}, 
                                "confidence": float(cf)})                      
            self.entry.setString(json.dumps(temp_entry))
          
            if self.fps % 100 == 0:
                print("Completed", self.fps, "frames. FPS:", (1 / (time.monotonic() - self.startTime)))

            # if self.fps % 10 == 0:
                # self.fps_entry.setNumber((1 / (time.monotonic() - self.startTime)))

            self.fps += 1    