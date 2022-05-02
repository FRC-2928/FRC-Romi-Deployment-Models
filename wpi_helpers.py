#!/usr/bin/env python3

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
import cv2
from networktables import NetworkTablesInstance

# Constants
FRAME_WIDTH = 416
FRAME_HEIGHT = 416

class ConfigParser:
    def __init__(self, config_path):
        self.team = -1

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
        self.counter = 0  
        self.startTime = time.monotonic()

    def put_data(self, boxes, confidence, label, fps):
        
        for bb, cf, cl in zip(boxes, confidence, label):
            temp_entry = []
            cls_name = self.labelMap.get(int(cl))
            xmin, ymin, xmax, ymax = bb[0], bb[1], bb[2], bb[3]
            temp_entry.append({"label": cls_name, 
                                "box": {"ymin": int(ymin), "xmin": int(xmin), "ymax": int(ymax), "xmax": int(xmax)}, 
                                "confidence": float(cf)})                      
            self.entry.setString(json.dumps(temp_entry))
            self.fps_entry.setNumber(fps)
            # self.fps_entry.setNumber((self.counter / (time.monotonic() - self.startTime)))
            self.counter += 1     