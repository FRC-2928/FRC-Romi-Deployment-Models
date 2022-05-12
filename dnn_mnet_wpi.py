'''
PRE-REQUISITES:

Install the packages:

pip install numpy
pip install argparse
pip install pillow

You may also need to install these if not deploying to the Romi:

pip install robotpy-cscore
pip install opencv-python==3.4.2

To run this on the Romi:

    python3 dnn_mnet_wpi.py

Output is displayed to http://wpilibpi.local:1182    

If you're on a PC you can use the OpenCV camera in GUI desktop window 

    python3 dnn_mnet_wpi.py --use_cv2_camera --frc_config frc.json

'''


# importing the necessary packages
import numpy as np
import argparse
import time
import cv2
import sys
from pathlib import Path
from wpi_helpers import ConfigParser, WPINetworkTables
# from cscore import CameraServer

def parse_args():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", action='store_true',
        help="path to input test image [False]")
    ap.set_defaults(image=False)   
    ap.add_argument("-m", "--model", type=str, required=False,
        default='rapid-react',
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

def main():
    args = parse_args()

    # Load the FRC configuration file
    config_file = args.frc_config
    config_parser = ConfigParser(config_file)

    # paths to the MobileNet model and configuration
    tflite_file = f"{args.model}.pb"
    pbtxt_file = f"{args.model}.pbtxt"
    tflitePath = str((Path(__file__).parent / Path(tflite_file)).resolve().absolute())
    pbtxtPath = str((Path(__file__).parent / Path(pbtxt_file)).resolve().absolute())
    print(f"Model file path {tflitePath}")

    # load our MobileNetSSD object detector 
    net = cv2.dnn.readNetFromTensorflow(tflitePath, pbtxtPath)

    # determine only the *output* layer names that we need from YOLO
    # ln = net.getLayerNames()
    # layerNames = [ln[i] for i in net.getUnconnectedOutLayers()]

    # Start the camera
    camera_config = config_parser.cameras[1]
    WIDTH, HEIGHT = camera_config["width"], camera_config["height"]
    img = np.zeros(shape=(WIDTH, HEIGHT, 3), dtype=np.uint8)

    if args.use_cv2_camera is True:
        # Use regular OpenCV camera
        camera = start_camera(WIDTH, HEIGHT)
        mjpegServer = False       
    else:    
        # Use robotpy-cscore camera server
        cs = start_cameraServer(WIDTH, HEIGHT)   
        camera = cs.getVideo()
        mjpegServer = cs.putVideo("OpenCV DNN", WIDTH, HEIGHT)
        

    try:
        run(camera, net, args, img=img, mjpegServer=mjpegServer)
    except Exception as e:
        print(e)
    finally:
        print("Done...")
        if args.use_cv2_camera is True:
            camera.release()

def run(camera,  net, args, img, mjpegServer):

    # load the COCO class labels our YOLO model was trained on
    labelsPath = f"{args.model}.names"
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    while True:

        if args.image is True:
            # loads a test input image     
            image = cv2.imread(args.image)   
        elif args.use_cv2_camera is True:   
            # read from the OpenCV camera
            success, image = camera.read()
            if not success:
                sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')            
        else:
            # read from robotpy-cscore camera server
            success, image = camera.grabFrame(img)   
            if not success:
                print("Image failed")
                continue    

        # Get its spatial dimensions     
        image = cv2.resize(image,(300,300)) # resize frame for prediction
        (H, W) = image.shape[:2]

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size. 
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
        # blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        #Set to network the input blob 
        net.setInput(blob)
        #Prediction of network
        detections = net.forward()

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # net.setInput(blob)
        # layerOutputs = net.forward(layerNames)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        # boxes = []
        # confidences = []
        # classIDs = []

        #Size of frame resize (300x300)
        # cols = image_resized.shape[1] 
        # rows = image_resized.shape[0]

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @detections array .
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] #Confidence of prediction 
            if confidence > args.threshold: # Filter prediction 
                class_id = int(detections[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(detections[0, 0, i, 3] * W) 
                yLeftBottom = int(detections[0, 0, i, 4] * H)
                xRightTop   = int(detections[0, 0, i, 5] * W)
                yRightTop   = int(detections[0, 0, i, 6] * H)
                # Factor for scale to original size of frame
                heightFactor = image.shape[0]/300.0  
                widthFactor = image.shape[1]/300.0 
                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom) 
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)
                # Draw location of object  
                cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                            (0, 255, 0))

                # Draw label and confidence of prediction in frame resized
                if class_id in LABELS:
                    label = LABELS[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                        (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # show the output image
        if mjpegServer is False:
            cv2.imshow("Image", image)
        else:
            mjpegServer.putFrame(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()