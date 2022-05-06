'''
PRE-REQUISITES:

Install the packages:

pip install opencv-python==3.4.2
pip install numpy
pip install argparse
pip install pillow
pip install robotpy-cscore

You can run the file by using 

    python3 dnn_yolo_wpi.py

To use the OpenCV camera

    python3 dnn_yolo_wpi.py --use_cv2_camera

'''


# importing the necessary packages
import numpy as np
import argparse
import time
import cv2
import sys
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
    
    # img = np.zeros(shape=(320, 512, 3), dtype=np.uint8)
    return cs

def start_camera(WIDTH, HEIGHT):
    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return camera    

def main():
    args = parse_args()

    # Load the FRC configuration file
    config_file = args.frc_config
    config_parser = ConfigParser(config_file)

    # paths to the YOLO weights and model configuration
    weightsPath = f"{args.model}.weights"
    configPath = f"{args.model}.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    layerNames = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

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
        run(camera, net, layerNames, args, img=img, mjpegServer=mjpegServer)
    except Exception as e:
        print(e)
    finally:
        print("Done...")
        if args.use_cv2_camera is True:
            camera.release()

def run(camera,  net, layerNames, args, img, mjpegServer):

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
        (H, W) = image.shape[:2] 
        print(image.shape[:2])

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(layerNames)

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    # print(f"appending {width} : {height}")
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence, args.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        # show the output image
        if mjpegServer is False:
            cv2.imshow("Image", image)
        else:
            mjpegServer.putFrame(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    main()